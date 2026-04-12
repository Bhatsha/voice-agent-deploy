"""
gemini_live_agent.py — Voice agent using Gemini Live API (audio-in / audio-out).

Drop-in replacement for VoiceAgent in agent.py. Same interface as called by main.py.
Architecture:
  Exotel (8kHz PCM, base64) → Gemini Live (audio/pcm;rate=8000)
  Gemini Live (24kHz PCM)   → ffmpeg (24kHz→8kHz) → base64 → Exotel WebSocket
"""

import asyncio
import base64
import logging
import subprocess
import time
from typing import Callable, Optional

import httpx
from google import genai
from google.genai import types

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tool definitions for Gemini function calling
# ---------------------------------------------------------------------------
_TOOLS = types.Tool(
    function_declarations=[
        types.FunctionDeclaration(
            name="set_call_status",
            description="Call immediately when the order outcome is determined (ACCEPTED, REJECTED, MODIFIED, or CALLBACK_REQUESTED).",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "status": types.Schema(
                        type=types.Type.STRING,
                        description="One of: ACCEPTED, REJECTED, MODIFIED, or CALLBACK_REQUESTED",
                    ),
                    "reason": types.Schema(
                        type=types.Type.STRING,
                        description="Reason for rejection or modification details (clear spoken Tamil)",
                    ),
                },
                required=["status"],
            ),
        ),
        types.FunctionDeclaration(
            name="end_call",
            description="Call after saying the farewell phrase to end the call.",
            parameters=types.Schema(
                type=types.Type.OBJECT,
                properties={},
            ),
        ),
    ]
)


def _build_system_instruction(order_data: dict) -> str:
    """Build the Gemini Live system instruction from order_data."""
    greeting = config.build_greeting(order_data)
    total = config._calc_total(order_data)
    total_word = config.amount_to_tamil(total)

    # Pull in the full rule-set from the existing system prompt builder
    full_prompt = config.build_system_prompt(order_data)

    return f"""You are Ramesh, a Tamil-speaking call executive from Keeggi confirming food orders with restaurant vendors.

CRITICAL: Start the conversation IMMEDIATELY by saying this exact greeting (nothing else first):
"{greeting}"

Do NOT wait for the user to speak first. Speak the greeting as your very first output.

LANGUAGE & SPEECH RULES:
- Speak ONLY in Tamil. Every word must be Tamil script.
- The ONLY English allowed: item names (Chicken Biryani, Sambar Rice, etc.) and "Order ID".
- Use spoken/colloquial Tamil — NOT written/formal Tamil.
- Keep responses to 1-2 SHORT sentences. Maximum 15 words per response.
- Use natural fillers: அப்போ..., சரி..., ஹ்ம்ம்..., ஓகே...
- When saying price/amount, ALWAYS say the amount in Tamil words followed by "ரூபாய்". NEVER use digits or "RS/rupees".
- Vary phrasing — do not repeat exact same sentences.

ORDER DETAILS:
- Order ID: {order_data['order_id']}
- Vendor: {order_data['vendor_name']}
- Company: {order_data['company_name']}
- Items: {', '.join(f"{i['name']} x{i['qty']}" for i in order_data['items'])}
- Total: {total_word} ரூபாய்

INTENT HANDLING (follow exactly):

1. ACCEPTED — vendor says சரி/ஓகே/confirm/yes/எடுத்துக்கலாம் (with NO change/modify mention):
   - Confirm with vendor once: "ஓகே, அப்போ ஆர்டர் எடுக்கிறீங்க, சரியா?"
   - On vendor's confirmation: call set_call_status(status="ACCEPTED")
   - Then say: "சரி, ஆர்டர் உறுதி பண்ணிட்டேன். வேற ஏதாவது இருக்கா?"
   - When vendor says no/nothing more: say farewell then call end_call()

2. REJECTED — vendor says வேணாம்/முடியாது/reject/cancel/இல்லை (to the ENTIRE order):
   - Ask gently: "சரி, வேண்டாம்-னா காரணம் சொல்லுங்க?"
   - Accept their reason, confirm: "சரி, [reason]-னால வேணாம்-னு சொல்றீங்க, சரியா?"
   - On confirmation: call set_call_status(status="REJECTED", reason="<reason in Tamil>")
   - Say: "சரி, புரிஞ்சது. வேற ஏதாவது இருக்கா?"
   - When vendor says no: say farewell then call end_call()

3. MODIFIED — vendor says modify/change/மாத்தணும்/one item unavailable but rest okay:
   - HIGHEST priority. Even "X மட்டும் இல்லை, மற்றது சரி" = MODIFICATION not REJECTION.
   - Respond: "சரி, மாற்றம் வேணும்-னா Keeggi-கிட்ட தொடர்பு கொள்ளுங்க. நன்றி."
   - call set_call_status(status="MODIFIED", reason="<what vendor wants changed in Tamil>")
   - Ask: "வேற ஏதாவது இருக்கா?"
   - When vendor says no: say farewell then call end_call()

4. CALLBACK_REQUESTED — vendor says call later/இப்போ முடியாது/later-ல call பண்ணுங்க:
   - Respond: "சரி, அப்புறம் call பண்றேன். நன்றி."
   - call set_call_status(status="CALLBACK_REQUESTED")
   - Say farewell then call end_call()

5. HOLD — vendor says ஒரு நிமிஷம்/hold/wait/busy:
   - Respond: "சரி, காத்திருக்கிறேன்... நிதானமா சொல்லுங்க."

6. PRICE QUERY — vendor asks price/total/எவ்வளவு/விலை:
   - Say only: "மொத்தம் {total_word} ரூபாய்." then ask "ஆர்டர் எடுத்துக்கலாமா?"

7. REPEAT/CLARIFY — vendor asks to repeat the order:
   - Repeat items with quantities in spoken Tamil, then ask "ஆர்டர் எடுத்துக்கலாமா?"

8. SILENCE / no response — say: "ஹலோ, கேட்கிறீங்களா?"

FAREWELL (use EXACTLY this phrase before calling end_call):
"சரி, நன்றி! நல்ல நாளா இருக்கட்டும்... வணக்கம்!"

RULES:
- NEVER mention price unless vendor asks.
- NEVER say "I am AI" or "I am a bot". Stay as Ramesh.
- Use vendor name ONLY in first reply.
- NEVER use "சார்" or "மேடம்".
- NEVER ask more than ONE question per reply.
- Always call set_call_status BEFORE asking "வேற ஏதாவது இருக்கா?"
- Always call end_call AFTER the farewell phrase, never before.
- If vendor changes mind (said yes then says no), always respect the LATEST intent.
"""


class GeminiLiveAgent:
    """
    Voice agent using Gemini Live API for end-to-end audio conversation.

    Interface matches VoiceAgent so main.py can use it as a drop-in replacement.
    """

    # Chunk size for reading resampled audio from ffmpeg stdout (320 bytes = 20ms @ 8kHz mono s16le)
    FFMPEG_READ_CHUNK = 320

    def __init__(
        self,
        exotel_ws,
        stream_sid: str,
        call_sid: str,
        order_data: dict,
        api_key: Optional[str] = None,
        on_key_release: Optional[Callable] = None,
    ):
        self.exotel_ws = exotel_ws
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        self.order_data = order_data
        self._api_key = api_key or config.GEMINI_API_KEY
        self._on_key_release = on_key_release

        # State
        self._call_ended = False
        self._webhook_sent = False
        self._final_status: Optional[str] = None
        self._final_reason: str = ""
        self._gemini_speaking = False
        self._last_audio_finished: float = 0.0

        # Gemini session (set in start())
        self._client: Optional[genai.Client] = None
        self._session = None
        self._live_ctx = None  # context manager reference for proper cleanup

        # Background tasks
        self._receive_task: Optional[asyncio.Task] = None
        self._ffmpeg_reader_task: Optional[asyncio.Task] = None

        # ffmpeg process for 24kHz → 8kHz resampling
        self._ffmpeg_proc: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    async def start(self):
        """Connect to Gemini Live and start background receive/ffmpeg tasks."""
        logger.info(f"GeminiLiveAgent starting for call {self.call_sid}")

        # Build Gemini client
        self._client = genai.Client(
            api_key=self._api_key,
            http_options={"api_version": "v1beta"},
        )

        system_instruction = _build_system_instruction(self.order_data)

        live_config = types.LiveConnectConfig(
            response_modalities=["AUDIO"],
            system_instruction=types.Content(
                parts=[types.Part(text=system_instruction)]
            ),
            tools=[_TOOLS],
        )

        # Start ffmpeg resample process
        self._start_ffmpeg()

        # Open persistent Gemini Live session
        try:
            self._live_ctx = self._client.aio.live.connect(
                model=config.GEMINI_TTS_MODEL,
                config=live_config,
            )
            self._session = await self._live_ctx.__aenter__()
        except Exception as e:
            logger.error(f"Failed to connect to Gemini Live: {e}")
            raise

        # Start background tasks
        self._receive_task = asyncio.create_task(self._receive_loop())
        self._ffmpeg_reader_task = asyncio.create_task(self._ffmpeg_reader_loop())

        logger.info(f"GeminiLiveAgent connected — call {self.call_sid}")

        # Trigger Gemini to speak the greeting immediately
        await asyncio.sleep(0.3)
        try:
            await self._session.send_realtime_input(text=".")
            logger.info("Greeting trigger sent to Gemini")
        except Exception as e:
            logger.error(f"Failed to send greeting trigger: {e}")

    async def handle_media(self, payload: str):
        """
        Receive base64-encoded 8kHz linear16 PCM from Exotel and forward to Gemini.
        Echo suppression: drop audio while Gemini is speaking or just finished.
        """
        if self._call_ended or self._session is None:
            return

        # Echo suppression
        if self._gemini_speaking:
            return
        if self._last_audio_finished > 0 and (time.time() - self._last_audio_finished) < 0.5:
            return

        try:
            audio_bytes = base64.b64decode(payload)
            await self._session.send_realtime_input(
                audio=types.Blob(data=audio_bytes, mime_type="audio/pcm;rate=8000")
            )
        except Exception as e:
            if not self._call_ended:
                logger.error(f"handle_media error: {e}")

    async def handle_flush(self):
        """No-op for Gemini Live (streaming is continuous)."""
        pass

    async def handle_dtmf(self, digit: str):
        """Handle DTMF digit — send as text input to Gemini."""
        if self._call_ended or self._session is None:
            return
        logger.info(f"DTMF digit: {digit}")
        try:
            await self._session.send_realtime_input(
                text=f"[DTMF: {digit}]"
            )
        except Exception as e:
            logger.error(f"handle_dtmf error: {e}")

    async def stop(self):
        """Clean up all resources."""
        logger.info(f"GeminiLiveAgent stopping for call {self.call_sid}")

        # If we have a final status but haven't sent the webhook, do it now
        if self._final_status and not self._webhook_sent:
            logger.info(f"CLEANUP: sending webhook for {self._final_status}")
            await self._send_webhook()

        # Cancel background tasks
        for task in (self._receive_task, self._ffmpeg_reader_task):
            if task and not task.done():
                task.cancel()
                try:
                    await task
                except (asyncio.CancelledError, Exception):
                    pass

        # Close ffmpeg
        self._stop_ffmpeg()

        # Close Gemini session
        if self._live_ctx is not None:
            try:
                await self._live_ctx.__aexit__(None, None, None)
            except Exception as e:
                logger.debug(f"Session close: {e}")
            self._session = None
            self._live_ctx = None

        # Release API key back to pool (no-arg closure from main.py)
        if self._on_key_release:
            try:
                self._on_key_release()
            except Exception:
                pass

        logger.info(f"GeminiLiveAgent stopped for call {self.call_sid}")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _start_ffmpeg(self):
        """Launch persistent ffmpeg process: 24kHz s16le → 8kHz s16le."""
        try:
            self._ffmpeg_proc = subprocess.Popen(
                [
                    "ffmpeg",
                    "-loglevel", "error",
                    "-f", "s16le", "-ar", "24000", "-ac", "1", "-i", "pipe:0",
                    "-f", "s16le", "-ar", "8000", "-ac", "1", "pipe:1",
                ],
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
            )
            logger.info("ffmpeg resample process started (24kHz→8kHz)")
        except FileNotFoundError:
            logger.error("ffmpeg not found — audio resampling will fail")
            self._ffmpeg_proc = None

    def _stop_ffmpeg(self):
        """Terminate ffmpeg process."""
        if self._ffmpeg_proc is not None:
            try:
                self._ffmpeg_proc.stdin.close()
            except Exception:
                pass
            try:
                self._ffmpeg_proc.terminate()
                self._ffmpeg_proc.wait(timeout=2)
            except Exception:
                pass
            self._ffmpeg_proc = None

    async def _feed_ffmpeg(self, pcm_24k: bytes):
        """Write 24kHz PCM data to ffmpeg stdin (non-blocking via executor)."""
        if self._ffmpeg_proc is None or self._ffmpeg_proc.stdin is None:
            return
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._ffmpeg_proc.stdin.write, pcm_24k)
        except (BrokenPipeError, OSError):
            pass
        except Exception as e:
            logger.error(f"ffmpeg feed error: {e}")

    async def _ffmpeg_reader_loop(self):
        """
        Background task: read resampled 8kHz PCM from ffmpeg stdout and
        forward as base64 media events to Exotel WebSocket.
        """
        loop = asyncio.get_event_loop()
        try:
            while not self._call_ended:
                if self._ffmpeg_proc is None or self._ffmpeg_proc.stdout is None:
                    await asyncio.sleep(0.02)
                    continue

                try:
                    chunk = await loop.run_in_executor(
                        None,
                        self._ffmpeg_proc.stdout.read,
                        self.FFMPEG_READ_CHUNK,
                    )
                except (OSError, ValueError):
                    break

                if not chunk:
                    await asyncio.sleep(0.005)
                    continue

                # Send to Exotel as base64 media event
                encoded = base64.b64encode(chunk).decode("utf-8")
                try:
                    await self.exotel_ws.send_json({
                        "event": "media",
                        "streamSid": self.stream_sid,
                        "media": {"payload": encoded},
                    })
                except Exception as e:
                    if not self._call_ended:
                        logger.error(f"Exotel send error: {e}")
                    break

        except asyncio.CancelledError:
            pass
        except Exception as e:
            logger.error(f"ffmpeg reader loop error: {e}")

    async def _receive_loop(self):
        """
        Background task: receive audio and tool-call responses from Gemini Live.
        Outer while loop ensures we re-enter receive() after each turn completes.
        """
        try:
            while not self._call_ended:
                async for response in self._session.receive():
                    if self._call_ended:
                        break

                    # ---- Audio output ----
                    if response.data:
                        self._gemini_speaking = True
                        await self._feed_ffmpeg(response.data)

                    # ---- Tool calls ----
                    if response.tool_call:
                        tool_responses = []
                        for fc in response.tool_call.function_calls:
                            logger.info(f"Tool call: {fc.name} args={dict(fc.args)}")

                            if fc.name == "set_call_status":
                                self._final_status = fc.args.get("status")
                                self._final_reason = fc.args.get("reason", "")
                                logger.info(
                                    f"Status set: {self._final_status} reason={self._final_reason!r}"
                                )
                            elif fc.name == "end_call":
                                # Schedule end_call as a task so we can respond to the tool first
                                asyncio.create_task(self._end_call())

                            tool_responses.append(
                                types.FunctionResponse(
                                    name=fc.name,
                                    id=fc.id,
                                    response={"result": "ok"},
                                )
                            )

                        # Always acknowledge tool calls
                        if tool_responses:
                            try:
                                await self._session.send_tool_response(
                                    function_responses=tool_responses
                                )
                            except Exception as e:
                                logger.error(f"Tool response error: {e}")

                    # ---- Turn complete ----
                    if response.server_content and response.server_content.turn_complete:
                        self._gemini_speaking = False
                        self._last_audio_finished = time.time()
                        logger.info("Turn complete — waiting for next input")
                        # Flush ffmpeg by sending a brief silence so buffered audio drains
                        if self._ffmpeg_proc and self._ffmpeg_proc.stdin:
                            try:
                                # 100ms of silence at 24kHz mono s16le = 4800 bytes
                                silence = b"\x00" * 4800
                                loop = asyncio.get_event_loop()
                                await loop.run_in_executor(
                                    None, self._ffmpeg_proc.stdin.write, silence
                                )
                            except Exception:
                                pass

        except asyncio.CancelledError:
            pass
        except Exception as e:
            if not self._call_ended:
                logger.error(f"Gemini receive loop error: {e}")

    async def _end_call(self):
        """Handle end_call tool: send webhook and hang up."""
        if self._call_ended:
            return
        self._call_ended = True

        logger.info(f"_end_call triggered — status={self._final_status}")

        # Give audio a moment to finish streaming to Exotel
        await asyncio.sleep(2)

        # Notify browser tester (if any)
        try:
            await self.exotel_ws.send_json({
                "event": "end_call",
                "status": self._final_status or "UNKNOWN",
                "message": f"Call ended — {self._final_status}",
            })
        except Exception:
            pass

        # Send webhook
        await self._send_webhook()

        # Hang up Exotel
        await self._hangup_exotel_call()

    async def _send_webhook(self):
        """POST order result to config.WEBHOOK_URL."""
        if self._webhook_sent:
            return
        self._webhook_sent = True

        if not config.WEBHOOK_URL:
            logger.warning("WEBHOOK_URL not configured — skipping webhook")
            return

        payload = {
            "order_id": self.order_data["order_id"],
            "vendor_name": self.order_data["vendor_name"],
            "company": self.order_data["company_name"],
            "total_amount": config._calc_total(self.order_data),
            "status": self._final_status or "UNKNOWN",
            "reason": self._final_reason,
            "call_sid": self.call_sid,
        }

        logger.info(f"Sending webhook to {config.WEBHOOK_URL}: {payload}")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(config.WEBHOOK_URL, json=payload)
                logger.info(f"Webhook response: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")

    async def _hangup_exotel_call(self):
        """
        Hang up the Exotel call by closing the WebSocket.
        Exotel advances to the next applet (Hangup) when the WebSocket closes.
        """
        if not self.call_sid or self.call_sid.startswith("test-"):
            return  # Skip for browser tester

        try:
            await self.exotel_ws.close()
            logger.info("Closed WebSocket — Exotel will hangup")
        except Exception as e:
            logger.error(f"WebSocket close error: {e}")
