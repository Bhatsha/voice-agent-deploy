import asyncio
import base64
import logging
import struct
import time
from typing import Callable, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# At 8kHz 16-bit mono: 16000 bytes per second
BYTES_PER_SECOND_8K = 16000
# Send chunks of ~20ms worth of audio
CHUNK_DURATION_MS = 20
CHUNK_BYTES = int(BYTES_PER_SECOND_8K * CHUNK_DURATION_MS / 1000)  # 320 bytes per chunk


class ElevenLabsTTS:
    """Text-to-Speech client for ElevenLabs using HTTP streaming API."""

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._api_key = api_key or config.ELEVENLABS_API_KEY
        self._voice_id = config.ELEVENLABS_VOICE_ID
        self._speaking = False
        self._connected = False

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        self._connected = True
        self._log("ElevenLabs TTS ready (HTTP mode)")

    async def speak(self, text: str):
        """Convert text to speech via HTTP streaming with real-time pacing."""
        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")

        url = f"https://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream"
        headers = {
            "xi-api-key": self._api_key,
            "Content-Type": "application/json",
        }
        payload = {
            "text": text,
            "model_id": config.ELEVENLABS_MODEL,
            "output_format": "pcm_16000",
            "voice_settings": {
                "stability": 0.5,
                "similarity_boost": 0.8,
                "style": 0.3,
                "use_speaker_boost": True,
            },
        }

        try:
            # Collect all PCM 16kHz audio first
            pcm_16k = b""
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        self._log(f"HTTP error {resp.status_code}: {error_text[:200]}")
                        self._speaking = False
                        return

                    async for chunk in resp.aiter_bytes():
                        if not self._speaking:
                            break
                        pcm_16k += chunk

            if not self._speaking:
                return

            # Ensure even byte count
            if len(pcm_16k) % 2 != 0:
                pcm_16k = pcm_16k[:-1]

            # Downsample 16kHz → 8kHz: take every other sample
            num_samples = len(pcm_16k) // 2
            samples = struct.unpack(f"<{num_samples}h", pcm_16k)
            downsampled = samples[::2]
            pcm_8k = struct.pack(f"<{len(downsampled)}h", *downsampled)

            # Send audio in real-time paced chunks
            start_time = time.monotonic()
            bytes_sent = 0

            for i in range(0, len(pcm_8k), CHUNK_BYTES):
                if not self._speaking:
                    break

                chunk = pcm_8k[i:i + CHUNK_BYTES]
                audio_b64 = base64.b64encode(chunk).decode("ascii")
                await self.on_audio(audio_b64)
                bytes_sent += len(chunk)

                # Pace: wait until real-time catches up
                expected_time = bytes_sent / BYTES_PER_SECOND_8K
                elapsed = time.monotonic() - start_time
                if expected_time > elapsed:
                    await asyncio.sleep(expected_time - elapsed)

            if self._speaking:
                self._speaking = False
                self._log("Finished speaking")
                if self.on_done:
                    await self.on_done()

        except Exception as e:
            self._log(f"Speak error: {e}")
            self._speaking = False

    async def stop(self):
        self._speaking = False

    async def close(self):
        self._speaking = False
        self._connected = False
        logger.info("ElevenLabs TTS closed")
