import asyncio
import base64
import logging
import time
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
    """Text-to-Speech client for ElevenLabs."""

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

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        self._connected = True
        self._log("ElevenLabs TTS ready")

    async def speak(self, text: str):
        """Start TTS — returns immediately, audio plays in background."""
        if self._playback_task and not self._playback_task.done():
            self._speaking = False
            self._playback_task.cancel()
            try:
                await self._playback_task
            except (asyncio.CancelledError, Exception):
                pass

        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")
        self._playback_task = asyncio.create_task(self._generate_and_play(text))

    async def _generate_and_play(self, text: str):
        """Fetch PCM audio from ElevenLabs and play with real-time pacing."""
        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL,
            "output_format": "pcm_8000",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True,
            },
        }

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(url, json=payload, headers=headers)

                if resp.status_code != 200:
                    self._log(f"HTTP error {resp.status_code}: {resp.text[:200]}")
                    self._speaking = False
                    return

                # pcm_8000 = raw linear16 PCM at 8kHz — exactly what Exotel expects
                pcm_audio = resp.content
                self._log(f"Got {len(pcm_audio)} bytes of PCM audio")

            if not self._speaking or not pcm_audio:
                return

            # Play audio in real-time paced chunks (no conversion needed)
            start_time = time.monotonic()
            bytes_sent = 0

            for i in range(0, len(pcm_audio), CHUNK_BYTES):
                if not self._speaking:
                    break

                chunk = pcm_audio[i:i + CHUNK_BYTES]
                audio_b64 = base64.b64encode(chunk).decode("ascii")
                await self.on_audio(audio_b64)
                bytes_sent += len(chunk)

                # Pace to real-time
                expected_time = bytes_sent / BYTES_PER_SECOND
                elapsed = time.monotonic() - start_time
                delay = expected_time - elapsed
                if delay > 0:
                    await asyncio.sleep(delay)

            if self._speaking:
                self._speaking = False
                self._log("Finished speaking")
                if self.on_done:
                    await self.on_done()

        except asyncio.CancelledError:
            self._speaking = False
        except Exception as e:
            self._log(f"Speak error: {e}")
            self._speaking = False

    async def stop(self):
        self._speaking = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()

    async def close(self):
        self._speaking = False
        self._connected = False
        if self._playback_task and not self._playback_task.done():
            self._playback_task.cancel()
        logger.info("ElevenLabs TTS closed")
