import asyncio
import base64
import logging
from typing import Callable, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# linear16 8kHz mono: 16000 bytes per second (2 bytes per sample)
BYTES_PER_SECOND = 16000
# Send 20ms chunks
CHUNK_DURATION_MS = 20
CHUNK_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_MS / 1000)  # 320 bytes


class ElevenLabsTTS:
    """Text-to-Speech client for ElevenLabs with real-time ffmpeg streaming."""

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._api_key = api_key or config.ELEVENLABS_API_KEY
        self._voice_id = config.ELEVENLABS_VOICE_ID
        self._speaking = False
        self._connected = False
        self._playback_task: Optional[asyncio.Task] = None
        self._ffmpeg_proc: Optional[asyncio.subprocess.Process] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        self._connected = True
        self._http_client = httpx.AsyncClient(timeout=30)
        self._log("ElevenLabs TTS ready (streaming mode)")

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
        """Kill any running ffmpeg process."""
        if self._ffmpeg_proc and self._ffmpeg_proc.returncode is None:
            try:
                self._ffmpeg_proc.kill()
                await self._ffmpeg_proc.wait()
            except Exception:
                pass
            self._ffmpeg_proc = None

    async def _stream_and_play(self, text: str):
        """Stream MP3 from ElevenLabs → ffmpeg → PCM 8kHz → Exotel."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL,
            "language_code": "ta",
            "output_format": "mp3_22050_32",
            "apply_text_normalization": "on",
            "seed": 42,
            "voice_settings": {
                "stability": 0.9,
                "similarity_boost": 0.9,
                "style": 0.0,
                "speed": 1.0,
                "use_speaker_boost": True,
            },
        }

        try:
            # Start ffmpeg: MP3 stdin → linear16 PCM 8kHz stdout
            self._ffmpeg_proc = await asyncio.create_subprocess_exec(
                "ffmpeg", "-i", "pipe:0",
                "-f", "s16le", "-ar", "8000", "-ac", "1",
                "-loglevel", "error",
                "pipe:1",
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            # Start reading PCM from ffmpeg stdout in parallel
            reader_task = asyncio.create_task(self._read_and_send_pcm())

            # Stream MP3 from ElevenLabs → ffmpeg stdin
            client = self._http_client or httpx.AsyncClient(timeout=30)
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    self._log(f"HTTP error {resp.status_code}: {error_text[:200]}")
                    self._speaking = False
                    await self._kill_ffmpeg()
                    reader_task.cancel()
                    return

                async for chunk in resp.aiter_bytes(chunk_size=4096):
                    if not self._speaking:
                        break
                    if chunk and self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                        try:
                            self._ffmpeg_proc.stdin.write(chunk)
                            await self._ffmpeg_proc.stdin.drain()
                        except (BrokenPipeError, ConnectionResetError):
                            break

            # Close ffmpeg stdin to signal EOF
            if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                try:
                    self._ffmpeg_proc.stdin.close()
                    await self._ffmpeg_proc.stdin.wait_closed()
                except Exception:
                    pass

            # Wait for reader to finish sending all PCM
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
        """Read PCM from ffmpeg stdout and send to Exotel."""
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

            # Send remaining buffer
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
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("ElevenLabs TTS closed")
