import asyncio
import base64
import json
import logging
from typing import Callable, Optional

import websockets

import config

logger = logging.getLogger(__name__)


class ElevenLabsTTS:
    """Streaming Text-to-Speech client for ElevenLabs."""

    MAX_CONNECT_RETRIES = 3

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._sample_rate = sample_rate or config.TTS_SAMPLE_RATE_TELEPHONY
        self._api_key = api_key or config.ELEVENLABS_API_KEY
        self._voice_id = config.ELEVENLABS_VOICE_ID
        self.ws = None
        self._listen_task: Optional[asyncio.Task] = None
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
        """Open WebSocket connection to ElevenLabs TTS."""
        model_id = config.ELEVENLABS_MODEL
        url = (
            f"wss://api.elevenlabs.io/v1/text-to-speech/{self._voice_id}/stream-input"
            f"?model_id={model_id}"
            f"&output_format=ulaw_8000"
        )
        headers = {"xi-api-key": self._api_key}

        for attempt in range(self.MAX_CONNECT_RETRIES):
            try:
                self.ws = await websockets.connect(
                    url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                )

                # Send initial config (BOS - beginning of stream)
                await self.ws.send(json.dumps({
                    "text": " ",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.8,
                        "style": 0.3,
                        "use_speaker_boost": True,
                    },
                    "generation_config": {
                        "chunk_length_schedule": [120, 160, 250, 290],
                    },
                    "xi_api_key": self._api_key,
                }))

                self._connected = True
                self._log("Connected to ElevenLabs TTS")
                self._listen_task = asyncio.create_task(self._listen())
                return
            except Exception as e:
                if attempt < self.MAX_CONNECT_RETRIES - 1:
                    self._log(f"Connect failed (attempt {attempt + 1}/{self.MAX_CONNECT_RETRIES}), retrying in 1s: {e}")
                    await asyncio.sleep(1)
                else:
                    self._log(f"Connect FAILED after {self.MAX_CONNECT_RETRIES} attempts: {e}")
                    self._connected = False

    async def speak(self, text: str):
        """Convert text to speech."""
        if not self._connected or not self.ws:
            self._log("Not connected, reconnecting...")
            await self.connect()
            if not self._connected:
                self._log("Reconnect failed, cannot speak")
                return

        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")

        try:
            # Send text
            await self.ws.send(json.dumps({
                "text": text,
                "try_trigger_generation": True,
            }))
            # Send empty string to flush/trigger generation
            await self.ws.send(json.dumps({
                "text": "",
            }))
        except websockets.exceptions.ConnectionClosed:
            self._log("Connection closed during speak, reconnecting...")
            self._connected = False
            self._speaking = False
            await self.connect()
            if self._connected:
                try:
                    self._speaking = True
                    await self.ws.send(json.dumps({
                        "text": text,
                        "try_trigger_generation": True,
                    }))
                    await self.ws.send(json.dumps({
                        "text": "",
                    }))
                except Exception as e:
                    self._log(f"Retry failed: {e}")
                    self._speaking = False

    async def stop(self):
        self._speaking = False

    async def _listen(self):
        """Listen for audio chunks from ElevenLabs."""
        try:
            async for message in self.ws:
                data = json.loads(message)

                if data.get("audio"):
                    # ElevenLabs sends base64-encoded audio
                    audio_b64 = data["audio"]
                    if audio_b64 and self._speaking:
                        await self.on_audio(audio_b64)

                if data.get("isFinal"):
                    self._speaking = False
                    self._log("Finished speaking")
                    if self.on_done:
                        await self.on_done()

                if data.get("error"):
                    self._log(f"Error: {data['error']}")
                    self._speaking = False

        except websockets.exceptions.ConnectionClosed as e:
            self._log(f"Connection closed: {e}")
        except Exception as e:
            self._log(f"Listen error: {e}")
        finally:
            self._connected = False

    async def close(self):
        self._speaking = False
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
        if self.ws:
            try:
                # Send EOS (end of stream)
                await self.ws.send(json.dumps({"text": ""}))
            except Exception:
                pass
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None
        logger.info("ElevenLabs TTS closed")
