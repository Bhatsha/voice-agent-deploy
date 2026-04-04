import asyncio
import base64
import logging
from typing import Callable, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# Exotel audio: linear16 8kHz mono → 16000 bytes/sec
BYTES_PER_SECOND = 16000
CHUNK_DURATION_MS = 20
CHUNK_BYTES = int(BYTES_PER_SECOND * CHUNK_DURATION_MS / 1000)  # 320 bytes

# Google Cloud TTS REST endpoint (supports API key auth)
GOOGLE_TTS_URL = "https://texttospeech.googleapis.com/v1/text:synthesize"


class GoogleTTS:
    """Text-to-Speech using Google Cloud TTS — drop-in replacement for SarvamTTS.
    Outputs LINEAR16 at 8kHz directly — no ffmpeg needed.
    """

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._api_key = api_key or config.GEMINI_API_KEY  # Same Google API key
        self._voice_name = config.GOOGLE_TTS_VOICE
        self._language_code = config.GOOGLE_TTS_LANGUAGE
        self._speaking = False
        self._connected = False
        self._speak_task: Optional[asyncio.Task] = None
        self._http_client: Optional[httpx.AsyncClient] = None

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[TTS] {msg}"))

    @property
    def is_speaking(self) -> bool:
        return self._speaking

    async def connect(self):
        self._http_client = httpx.AsyncClient(timeout=15)
        self._connected = True
        self._log(f"Google TTS ready (voice={self._voice_name})")

    async def speak(self, text: str):
        """Convert text to speech — returns immediately, streams in background."""
        if self._speak_task and not self._speak_task.done():
            self._speaking = False
            self._speak_task.cancel()
            try:
                await self._speak_task
            except (asyncio.CancelledError, Exception):
                pass

        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")
        self._speak_task = asyncio.create_task(self._fetch_and_stream(text))

    async def _fetch_and_stream(self, text: str):
        """Fetch audio from Google TTS and stream 20ms chunks to Exotel."""
        try:
            payload = {
                "input": {"text": text},
                "voice": {
                    "languageCode": self._language_code,
                    "name": self._voice_name,
                },
                "audioConfig": {
                    "audioEncoding": "LINEAR16",
                    "sampleRateHertz": 8000,
                    "speakingRate": config.GOOGLE_TTS_SPEED,
                },
            }

            client = self._http_client or httpx.AsyncClient(timeout=15)
            resp = await client.post(
                GOOGLE_TTS_URL,
                params={"key": self._api_key},
                json=payload,
            )

            if resp.status_code != 200:
                self._log(f"HTTP error {resp.status_code}: {resp.text[:200]}")
                self._speaking = False
                return

            audio_b64 = resp.json().get("audioContent", "")
            if not audio_b64:
                self._log("Empty audio response")
                self._speaking = False
                return

            audio_bytes = base64.b64decode(audio_b64)

            # Stream in 20ms chunks
            for i in range(0, len(audio_bytes), CHUNK_BYTES):
                if not self._speaking:
                    break
                chunk = audio_bytes[i:i + CHUNK_BYTES]
                await self.on_audio(base64.b64encode(chunk).decode("ascii"))
                # Pace the chunks to simulate real-time streaming
                await asyncio.sleep(CHUNK_DURATION_MS / 1000)

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
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()

    async def close(self):
        self._speaking = False
        self._connected = False
        if self._speak_task and not self._speak_task.done():
            self._speak_task.cancel()
        if self._http_client:
            await self._http_client.aclose()
            self._http_client = None
        logger.info("Google TTS closed")
