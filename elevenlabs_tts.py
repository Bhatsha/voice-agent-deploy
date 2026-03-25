import asyncio
import audioop
import base64
import logging
import struct
from typing import Callable, Optional

import httpx

import config

logger = logging.getLogger(__name__)


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
        self._speak_task: Optional[asyncio.Task] = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        """Mark as ready (HTTP doesn't need persistent connection)."""
        self._connected = True
        self._log("ElevenLabs TTS ready (HTTP mode)")

    async def speak(self, text: str):
        """Convert text to speech via HTTP streaming."""
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
            async with httpx.AsyncClient(timeout=30) as client:
                async with client.stream("POST", url, json=payload, headers=headers) as resp:
                    if resp.status_code != 200:
                        error_text = await resp.aread()
                        self._log(f"HTTP error {resp.status_code}: {error_text[:200]}")
                        self._speaking = False
                        return

                    async for chunk in resp.aiter_bytes(chunk_size=4096):
                        if not self._speaking:
                            break
                        if chunk:
                            # ElevenLabs sends PCM 16-bit 16kHz
                            # Downsample 16kHz → 8kHz for Exotel (linear16 8kHz)
                            pcm_8k = audioop.ratecv(chunk, 2, 1, 16000, 8000, None)[0]
                            audio_b64 = base64.b64encode(pcm_8k).decode("ascii")
                            await self.on_audio(audio_b64)

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
