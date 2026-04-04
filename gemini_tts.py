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


class GeminiTTS:
    """Text-to-Speech using Gemini Live API with persistent connection.
    One WebSocket connection per call — eliminates per-utterance setup overhead.
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
        self._session = None
        self._live_ctx = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        """Open one persistent Live API connection for the entire call."""
        from google import genai
        from google.genai import types

        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1beta"},
        )

        live_config = types.LiveConnectConfig(
            system_instruction=types.Content(
                parts=[types.Part(text=_TTS_SYSTEM_INSTRUCTION)]
            ),
            response_modalities=["AUDIO"],
        )

        self._live_ctx = self._client.aio.live.connect(
            model=self._model, config=live_config
        )
        self._session = await self._live_ctx.__aenter__()
        self._connected = True
        self._log(f"Gemini Live TTS connected (model={self._model})")

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
        """Send text on persistent session, stream PCM 24kHz → ffmpeg → 8kHz → Exotel."""
        try:
            if not self._session:
                self._log("No session — reconnecting")
                await self.connect()

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

            await self._session.send_realtime_input(text=text)

            async for msg in self._session.receive():
                if not self._speaking:
                    break

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
            # Reconnect for next speak()
            self._session = None

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
        if self._live_ctx and self._session:
            try:
                await self._live_ctx.__aexit__(None, None, None)
            except Exception:
                pass
        self._session = None
        self._live_ctx = None
        logger.info("Gemini TTS closed")
