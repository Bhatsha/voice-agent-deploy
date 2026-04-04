import asyncio
import base64
import logging
from typing import Callable, Optional

import config

logger = logging.getLogger(__name__)

# Exotel audio: linear16 8kHz mono → 16000 bytes/sec
BYTES_PER_SECOND = 16000
CHUNK_DURATION_MS = 20
CHUNK_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_MS / 1000)  # 320 bytes

_TTS_SYSTEM_INSTRUCTION = (
    "You are a text-to-speech reader. Speak exactly the text the user sends, "
    "word for word, in the same language. Do not add, change, or omit any words."
)


def _parse_sample_rate(mime_type: str) -> int:
    """Extract sample rate from mime type like 'audio/pcm;rate=24000'."""
    for part in (mime_type or "").split(";"):
        part = part.strip()
        if part.lower().startswith("rate="):
            try:
                return int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
    return 24000


class GeminiTTS:
    """Text-to-Speech using Gemini Live API (gemini-3.1-flash-live-preview).
    True streaming — first audio chunk arrives in ~1.8s.
    Drop-in replacement for ElevenLabsTTS / SarvamTTS.
    """

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._api_key = api_key or config.GEMINI_API_KEY
        self._model = config.GEMINI_TTS_MODEL
        self._speaking = False
        self._connected = False
        self._playback_task: Optional[asyncio.Task] = None
        self._ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self._client = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        from google import genai
        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1beta"},
        )
        self._connected = True
        self._log(f"Gemini Live TTS ready (model={self._model})")

    async def speak(self, text: str):
        """Start TTS — returns immediately, audio streams in background."""
        if self._playback_task and not self._playback_task.done():
            self._speaking = False
            await self._kill_ffmpeg()
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass

        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")
        self._playback_task = asyncio.create_task(self._stream_and_play(text))

    async def _kill_ffmpeg(self):
        if self._ffmpeg_proc and self._ffmpeg_proc.returncode is None:
            try:
                self._ffmpeg_proc.kill()
                await self._ffmpeg_proc.wait()
            except Exception:
                pass
            self._ffmpeg_proc = None

    async def _stream_and_play(self, text: str):
        """Connect to Gemini Live, stream PCM 24kHz → ffmpeg → 8kHz → Exotel."""
        from google.genai import types

        try:
            live_config = types.LiveConnectConfig(
                system_instruction=types.Content(
                    parts=[types.Part(text=_TTS_SYSTEM_INSTRUCTION)]
                ),
                response_modalities=["AUDIO"],
            )

            # Start ffmpeg: PCM 24kHz → PCM 8kHz
            self._ffmpeg_proc = await asyncio.create_subprocess_exec(
                "ffmpeg",
                "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-loglevel", "error",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            reader_task = asyncio.create_task(self._read_and_send_pcm())

            async with self._client.aio.live.connect(
                model=self._model, config=live_config
            ) as session:
                await session.send_realtime_input(text=text)

                async for msg in session.receive():
                    if not self._speaking:
                        break

                    # Audio arrives as inline_data in server_content parts
                    if (msg.server_content and
                            msg.server_content.model_turn and
                            msg.server_content.model_turn.parts):
                        for part in msg.server_content.model_turn.parts:
                            if part.inline_data and part.inline_data.data:
                                raw = part.inline_data.data
                                if isinstance(raw, str):
                                    raw = base64.b64decode(raw)
                                if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                                    try:
                                        self._ffmpeg_proc.stdin.write(raw)
                                        await self._ffmpeg_proc.stdin.drain()
                                    except (BrokenPipeError, ConnectionResetError):
                                        break

                    if msg.server_content and msg.server_content.turn_complete:
                        break

            # EOF to ffmpeg
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                    await self._ffmpeg_proc.stdin.wait_closed()
                except Exception:
                    pass

            await reader_task

            if self._speaking:
                self._speaking = False
                self._log("Finished speaking")
                if self.on_done:
                    await self.on_done()

        except asyncio.CancelledError:
            self._speaking = False
            await self._kill_ffmpeg()
        except Exception as e:
            self._log(f"Speak error: {e}")
            self._speaking = False
            await self._kill_ffmpeg()

    async def _read_and_send_pcm(self):
        """Read resampled PCM from ffmpeg and send to Exotel in 20ms chunks."""
        buffer = b""
        try:
            while self._speaking and self._ffmpeg_proc and self._ffmpeg_proc.stdout:
                chunk = await self._ffmpeg_proc.stdout.read(CHUNK_BYTES)
                if not chunk:
                    break
                buffer += chunk
                while len(buffer) >= CHUNK_BYTES and self._speaking:
                    pcm_chunk = buffer[:CHUNK_BYTES]
                    buffer = buffer[CHUNK_BYTES:]
                    await self.on_audio(base64.b64encode(pcm_chunk).decode("ascii"))
            if buffer and self._speaking:
                await self.on_audio(base64.b64encode(buffer).decode("ascii"))
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self._log(f"PCM reader error: {e}")

    async def stop(self):
        self._speaking = False
        await self._kill_ffmpeg()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

    async def close(self):
        self._speaking = False
        self._connected = False
        await self._kill_ffmpeg()
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        logger.info("Gemini TTS closed")
