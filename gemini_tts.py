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


def _parse_sample_rate(mime_type: str) -> int:
    """Extract sample rate from MIME type like 'audio/L16;rate=24000'."""
    for part in mime_type.split(";"):
        part = part.strip()
        if part.lower().startswith("rate="):
            try:
                return int(part.split("=", 1)[1])
            except (ValueError, IndexError):
                pass
    return 24000  # Gemini TTS default


class GeminiTTS:
    """Text-to-Speech using Gemini 2.5 Flash TTS — drop-in replacement for ElevenLabsTTS."""

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._api_key = api_key or config.GEMINI_API_KEY
        self._voice_name = config.GEMINI_TTS_VOICE
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
        self._client = genai.Client(api_key=self._api_key)
        self._connected = True
        self._log(f"Gemini TTS ready (model={self._model}, voice={self._voice_name})")

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
        """Stream PCM from Gemini TTS → ffmpeg resample → Exotel 8kHz."""
        from google.genai import types

        try:
            contents = [
                types.Content(
                    role="user",
                    parts=[types.Part.from_text(text=text)],
                )
            ]
            generate_config = types.GenerateContentConfig(
                temperature=1,
                response_modalities=["audio"],
                speech_config=types.SpeechConfig(
                    voice_config=types.VoiceConfig(
                        prebuilt_voice_config=types.PrebuiltVoiceConfig(
                            voice_name=self._voice_name
                        )
                    )
                ),
            )

            # ffmpeg will be started on the first chunk (once we know sample rate)
            reader_task = None
            sample_rate_detected = False

            async for chunk in await self._client.aio.models.generate_content_stream(
                model=self._model,
                contents=contents,
                config=generate_config,
            ):
                if not self._speaking:
                    break
                if chunk.parts is None:
                    continue

                part = chunk.parts[0]
                if not (part.inline_data and part.inline_data.data):
                    continue

                inline_data = part.inline_data
                audio_bytes = inline_data.data

                # Start ffmpeg on first audio chunk using the detected sample rate
                if not sample_rate_detected:
                    sample_rate = _parse_sample_rate(inline_data.mime_type or "audio/L16;rate=24000")
                    sample_rate_detected = True
                    self._ffmpeg_proc = await asyncio.create_subprocess_exec(
                        "ffmpeg",
                        "-f", "s16le", "-ar", str(sample_rate), "-ac", "1",
                        "-i", "pipe:0",
                        "-f", "s16le", "-ar", "8000", "-ac", "1",
                        "-loglevel", "error",
                        "pipe:1",
                        stdin=asyncio.subprocess.PIPE,
                        stdout=asyncio.subprocess.PIPE,
                        stderr=asyncio.subprocess.PIPE,
                    )
                    reader_task = asyncio.create_task(self._read_and_send_pcm())
                    self._log(f"ffmpeg started (input {sample_rate}Hz → 8000Hz)")

                if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                    try:
                        self._ffmpeg_proc.stdin.write(audio_bytes)
                        await self._ffmpeg_proc.stdin.drain()
                    except (BrokenPipeError, ConnectionResetError):
                        break

            # Signal EOF to ffmpeg
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                    await self._ffmpeg_proc.stdin.wait_closed()
                except Exception:
                    pass

            if reader_task:
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
        """Read resampled PCM from ffmpeg stdout and send to Exotel in 20ms chunks."""
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
                    audio_b64 = base64.b64encode(pcm_chunk).decode("ascii")
                    await self.on_audio(audio_b64)
            # Flush remaining
            if buffer and self._speaking:
                audio_b64 = base64.b64encode(buffer).decode("ascii")
                await self.on_audio(audio_b64)
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
