import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

import config

logger = logging.getLogger(__name__)


class SarvamTTS:
    """Streaming Text-to-Speech client for Sarvam AI."""

    MAX_CONNECT_RETRIES = 3

    def __init__(self, on_audio: Callable, on_log: Callable = None, on_done: Callable = None,
                 codec: str = None, sample_rate: int = None, api_key: str = None):
        self.on_audio = on_audio
        self.on_log = on_log
        self.on_done = on_done
        self._codec = codec or config.TTS_CODEC
        self._sample_rate = sample_rate or config.TTS_SAMPLE_RATE
        self._api_key = api_key or config.SARVAM_API_KEY
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
        """Open WebSocket connection to Sarvam TTS and send config."""
        params = f"?model={config.TTS_MODEL}&send_completion_event=true"
        headers = {"Api-Subscription-Key": self._api_key}

        for attempt in range(self.MAX_CONNECT_RETRIES):
            try:
                self.ws = await websockets.connect(
                    config.SARVAM_TTS_WS + params,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                )

                await self.ws.send(json.dumps({
                    "type": "config",
                    "data": {
                        "target_language_code": config.LANGUAGE,
                        "speaker": config.SPEAKER,
                        "model": config.TTS_MODEL,
                        "pace": config.TTS_PACE,
                        "speech_sample_rate": self._sample_rate,
                        "output_audio_codec": self._codec,
                        "enable_preprocessing": True,
                    }
                }))

                self._connected = True
                self._log("Connected to Sarvam TTS")
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
        # Reconnect if needed
        if not self._connected or not self.ws:
            self._log("Not connected, reconnecting...")
            await self.connect()
            if not self._connected:
                self._log("Reconnect failed, cannot speak")
                return

        self._speaking = True
        self._log(f"Speaking: {text[:60]}...")

        try:
            await self.ws.send(json.dumps({
                "type": "text",
                "data": {"text": text}
            }))
            await self.ws.send(json.dumps({"type": "flush"}))
        except websockets.exceptions.ConnectionClosed:
            self._log("Connection closed during speak, reconnecting...")
            self._connected = False
            self._speaking = False
            await self.connect()
            if self._connected:
                try:
                    self._speaking = True
                    await self.ws.send(json.dumps({
                        "type": "text",
                        "data": {"text": text}
                    }))
                    await self.ws.send(json.dumps({"type": "flush"}))
                except Exception as e:
                    self._log(f"Retry failed: {e}")
                    self._speaking = False

    async def stop(self):
        self._speaking = False

    async def _listen(self):
        """Listen for audio chunks from Sarvam TTS."""
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "audio":
                    audio_b64 = data.get("data", {}).get("audio", "")
                    if audio_b64 and self._speaking:
                        await self.on_audio(audio_b64)

                elif msg_type == "event":
                    event_type = data.get("data", {}).get("event_type", "")
                    if event_type == "final":
                        self._speaking = False
                        self._log("Finished speaking")
                        if self.on_done:
                            await self.on_done()

                elif msg_type == "error":
                    err_msg = data.get("data", {}).get("message", str(data))
                    self._log(f"Error: {err_msg}")
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
                await self.ws.close()
            except Exception:
                pass
        self.ws = None
        logger.info("Sarvam TTS closed")
