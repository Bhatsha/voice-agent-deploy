import asyncio
import json
import logging
from typing import Callable, Optional

import websockets

import config

logger = logging.getLogger(__name__)


class SarvamSTT:
    """Streaming Speech-to-Text client for Sarvam AI (Saaras v3)."""

    MAX_RECONNECT = 3
    MAX_CONNECT_RETRIES = 3

    def __init__(self, on_transcript: Callable, on_log: Callable = None, on_vad: Callable = None,
                 api_key: str = None):
        self.on_transcript = on_transcript
        self.on_log = on_log
        self.on_vad = on_vad  # Called with ("speech_start" | "speech_end")
        self._api_key = api_key or config.SARVAM_API_KEY
        self.ws = None
        self._listen_task: Optional[asyncio.Task] = None
        self._connected = False
        self._chunks_sent = 0
        self._reconnect_attempts = 0
        self._should_run = True  # Set to False on intentional close

    def _log(self, msg):
        logger.info(msg)
        if self.on_log:
            asyncio.ensure_future(self.on_log(f"[STT] {msg}"))

    async def connect(self):
        """Open WebSocket connection to Sarvam STT (Saaras v3 with VAD)."""
        params = (
            f"?language-code={config.LANGUAGE}"
            f"&model={config.STT_MODEL}"
            f"&mode=transcribe"
            f"&sample_rate={config.SAMPLE_RATE}"
            f"&input_audio_codec=pcm_s16le"
            f"&vad_signals=true"
            f"&high_vad_sensitivity=true"
            f"&flush_signal=true"
        )
        headers = {"Api-Subscription-Key": self._api_key}

        for attempt in range(self.MAX_CONNECT_RETRIES):
            try:
                self._log(f"Connecting to {config.SARVAM_STT_WS} (model={config.STT_MODEL})...")
                self.ws = await websockets.connect(
                    config.SARVAM_STT_WS + params,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                )
                self._connected = True
                self._chunks_sent = 0
                self._reconnect_attempts = 0
                self._log("Connected OK (Saaras v3 + VAD)")
                self._listen_task = asyncio.create_task(self._listen())
                return
            except Exception as e:
                if attempt < self.MAX_CONNECT_RETRIES - 1:
                    self._log(f"Connect failed (attempt {attempt + 1}/{self.MAX_CONNECT_RETRIES}), retrying in 1s: {e}")
                    await asyncio.sleep(1)
                else:
                    self._log(f"Connect FAILED after {self.MAX_CONNECT_RETRIES} attempts: {e}")
                    self._connected = False

    async def send_audio(self, audio_base64: str):
        """Send base64-encoded PCM audio chunk to STT."""
        if not self._connected or not self.ws:
            return False
        try:
            await self.ws.send(json.dumps({
                "audio": {
                    "data": audio_base64,
                    "encoding": "audio/wav",
                    "sample_rate": config.SAMPLE_RATE,
                }
            }))
            self._chunks_sent += 1
            if self._chunks_sent == 1:
                self._log("First audio chunk sent OK")
            return True
        except websockets.exceptions.ConnectionClosed as e:
            self._log(f"Send failed - connection closed: {e}")
            self._connected = False
            return False
        except Exception as e:
            self._log(f"Send failed: {e}")
            return False

    async def flush(self):
        """Flush the STT buffer to get final transcript."""
        if not self._connected or not self.ws:
            return
        try:
            self._log("Flushing STT buffer...")
            await self.ws.send(json.dumps({"type": "flush"}))
        except Exception as e:
            self._log(f"Flush failed: {e}")

    async def _listen(self):
        """Listen for transcription results and VAD events from Sarvam STT."""
        self._log("Listener started (Saaras v3 with VAD signals)...")
        try:
            async for message in self.ws:
                data = json.loads(message)
                msg_type = data.get("type", "")

                if msg_type == "data":
                    inner = data.get("data", {})
                    transcript = inner.get("transcript", "")
                    if transcript:
                        self._log(f"TRANSCRIPT: {transcript}")
                        await self.on_transcript(transcript, True)

                elif msg_type == "events":
                    inner = data.get("data", {})
                    signal = inner.get("signal_type", "")
                    self._log(f"VAD: {signal}")
                    if self.on_vad and signal.upper() in ("START_SPEECH", "END_SPEECH"):
                        # Normalize to lowercase format expected by agent
                        normalized = "speech_start" if "START" in signal.upper() else "speech_end"
                        await self.on_vad(normalized)

                elif msg_type == "error":
                    err = data.get("data", {})
                    self._log(f"ERROR from Sarvam: {err}")

                else:
                    # Fallback format
                    transcript = data.get("transcript", "")
                    is_final = data.get("is_final", False)
                    if transcript:
                        self._log(f"TRANSCRIPT (fallback): {transcript} (final={is_final})")
                        await self.on_transcript(transcript, is_final)

        except websockets.exceptions.ConnectionClosed as e:
            self._log(f"Listener: connection closed: code={e.code} reason={e.reason}")
        except Exception as e:
            self._log(f"Listener error: {e}")
        finally:
            self._log(f"Listener stopped. Total chunks sent: {self._chunks_sent}")
            self._connected = False
            # Auto-reconnect if call still active
            if self._should_run and self._reconnect_attempts < self.MAX_RECONNECT:
                self._reconnect_attempts += 1
                self._log(f"Auto-reconnecting STT (attempt {self._reconnect_attempts}/{self.MAX_RECONNECT})...")
                await asyncio.sleep(1)
                await self.connect()

    async def close(self):
        """Close the STT connection."""
        self._should_run = False
        self._connected = False
        if self._listen_task:
            self._listen_task.cancel()
        if self.ws:
            try:
                await self.ws.close()
            except Exception:
                pass
        self.ws = None
        logger.info("Sarvam STT closed")
