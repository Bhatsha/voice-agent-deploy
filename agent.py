import asyncio
import json
import logging
import random
import re
from typing import Callable, Optional

import httpx
from starlette.websockets import WebSocket

import config
from sarvam_stt import SarvamSTT
from sarvam_tts import SarvamTTS
from sarvam_llm import SarvamLLM

logger = logging.getLogger(__name__)


class VoiceAgent:
    """Orchestrates a single call session: Exotel <-> STT <-> LLM <-> TTS."""

    ECHO_BUFFER_MS = 500
    FLUSH_AFTER_MS = 2000
    ECHO_WORD_OVERLAP_THRESHOLD = 0.5
    ECHO_MIN_WORDS = 4

    MIN_TTS_DURATION_FOR_SILENCE_TIMEOUT = 1.2  # seconds

    def __init__(self, exotel_ws: WebSocket, stream_sid: str, call_sid: str,
                 order_data: Optional[dict] = None,
                 api_key: str = None, on_key_release: Callable = None):
        self.exotel_ws = exotel_ws
        self.stream_sid = stream_sid
        self.call_sid = call_sid
        if not order_data:
            raise ValueError("No order data provided — cannot start call without order")
        self.order_data = order_data
        self._api_key = api_key
        self._on_key_release = on_key_release
        self._key_released = False
        self._greeting_fallback_task: Optional[asyncio.Task] = None

        self.stt: Optional[SarvamSTT] = None
        self.tts: Optional[SarvamTTS] = None
        self.llm: Optional[SarvamLLM] = None

        self._processing = False
        self._audio_chunks_received = 0
        self._audio_chunks_sent_to_stt = 0
        self._call_ended = False
        self._webhook_sent = False

        # Echo suppression tracking
        self._tts_last_finished = 0  # timestamp when TTS last finished generating
        self._last_audio_sent_time = 0  # timestamp of last audio chunk sent to STT
        self._flush_task: Optional[asyncio.Task] = None

        # Track last agent response for echo detection
        self._last_agent_text = ""

        # Queued transcript: saved when _processing is True, processed after current LLM finishes
        self._queued_transcript: Optional[str] = None

        # Track TTS audio duration for silence timeout
        self._tts_audio_bytes = 0  # total audio bytes for current utterance
        self._tts_speak_start = 0.0  # when speak() was called
        self._remaining_playback_sec = 0.0  # estimated remaining playback time

        # Silence timeout: re-prompt if no user response
        self._silence_timeout_task: Optional[asyncio.Task] = None
        self._silence_timeout_sec = 7  # seconds to wait after audio finishes playing
        self._silence_prompts_sent = 0  # track how many silence prompts we've sent
        self._max_silence_prompts = 2  # max re-prompts before giving up

        # Terminal status values that end the call
        self._terminal_statuses = [
            "ACCEPTED", "REJECTED", "MODIFIED", "CALLBACK_REQUESTED",
            "NO_RESPONSE",
        ]
        self._unclear_count = 0
        self._max_unclear_before_end = 3
        self._rejection_reason = ""
        self._modification_reason = ""
        self._confirmation_pending = None  # None or expected status: "ACCEPTED", "REJECTED", "MODIFIED"

        # VAD speech duration tracking for noise filtering
        self._vad_speech_start_time = 0.0  # timestamp when VAD detected speech start
        self._last_speech_duration_ms = 0.0  # duration of last speech segment in ms
        self.MIN_SPEECH_DURATION_MS = 200  # ignore speech shorter than this

        # Two-phase greeting: intro first, wait for any vendor response, then items
        self._greeting_phase = 0  # 0=normal, 1=waiting for vendor ack after intro
        self._intro_ack_event = asyncio.Event()

        self._acceptance_keywords = [
            "ஓகே", "ஓகே தான்", "ஓகேதான்", "சரி", "சரிங்க", "சரி தான்",
            "ஆமா", "ஆமாங்க", "okay", "ok", "yes", "accept", "ready",
            "எடுத்துக்கலாம்", "எடுத்துக்குறேன்", "ஏத்துக்குறேன்",
            "ஏத்துக்கிறேன்", "வாங்கிக்கலாம்", "அக்செப்ட்",
            "ரெடி", "கொடுங்க", "வாங்கலாம்", "கன்ஃபர்ம்", "confirm",
            "போடலாம்", "எடுத்துக்கிறேன்",
        ]
        self._acceptance_closings = [
            "சரிங்க, ஆர்டர் கன்ஃபர்ம் ஆயிடுச்சு. தேங்க்ஸ்!",
            "ஓகே, ஆர்டர் ஏத்துக்கிட்டோம். Thanks!",
            "சரி, accepted ஆயிடுச்சு. Bye!",
            "நல்லது, confirm பண்ணிட்டேன். வணக்கம்!",
        ]

    async def _send_log(self, msg: str):
        """Send a log event back to the browser."""
        try:
            await self.exotel_ws.send_json({
                "event": "log",
                "message": msg,
            })
        except Exception:
            pass

    async def start(self):
        """Initialize all AI services and send greeting."""
        await self._send_log("Starting voice agent...")

        self.llm = SarvamLLM(system_prompt=config.build_system_prompt(self.order_data))
        self.stt = SarvamSTT(
            on_transcript=self._on_transcript, on_log=self._send_log,
            on_vad=self._on_vad, api_key=self._api_key,
        )

        # Use telephony codec for real Exotel calls, mp3 for browser tester
        is_real_call = not self.call_sid.startswith("test-")
        if is_real_call:
            self.tts = SarvamTTS(
                on_audio=self._on_tts_audio, on_log=self._send_log, on_done=self._on_tts_done,
                codec=config.TTS_CODEC_TELEPHONY, sample_rate=config.TTS_SAMPLE_RATE_TELEPHONY,
                api_key=self._api_key,
            )
        else:
            self.tts = SarvamTTS(
                on_audio=self._on_tts_audio, on_log=self._send_log, on_done=self._on_tts_done,
                api_key=self._api_key,
            )

        await self.stt.connect()
        await self.tts.connect()

        # Graceful failure: if both STT and TTS failed to connect, end call immediately
        if not self.stt._connected and not self.tts._connected:
            await self._send_log("FATAL: Both STT and TTS failed to connect — ending call")
            await self._send_webhook("NO_RESPONSE")
            self._call_ended = True
            self._release_key()
            return

        await self._send_log(f"Agent ready for call {self.call_sid}")

        # Brief pause before speaking, then say hello
        await asyncio.sleep(0.5)
        await self._speak("ஹலோ")
        await asyncio.sleep(0.5)

        # Phase 1: Short intro (name + company + "new order")
        intro = config.build_greeting_intro(self.order_data)
        self._last_agent_text = intro
        self._greeting_phase = 1
        await self._speak(intro)

        # Wait for TTS to finish generating the intro
        while self.tts and self.tts.is_speaking:
            await asyncio.sleep(0.1)

        # Wait for vendor to respond (any response) or timeout after playback + 1.5s
        wait_time = self._remaining_playback_sec + 1.5
        try:
            await asyncio.wait_for(self._intro_ack_event.wait(), timeout=wait_time)
            await self._send_log("Vendor acknowledged intro — reading items")
        except asyncio.TimeoutError:
            await self._send_log("No response to intro — continuing with items")

        self._greeting_phase = 0

        # Phase 2: Read order items + total + "okay?"
        items_greeting = config.build_greeting_items(self.order_data)
        self._last_agent_text = items_greeting
        await self._speak(items_greeting)
        # Silence timeout will start when TTS finishes (in _on_tts_done)

        # Fallback: if _on_tts_done never fires (TTS broken), start silence timeout after 15s
        self._greeting_fallback_task = asyncio.create_task(self._greeting_fallback_timeout())

    async def handle_media(self, payload: str):
        """Handle incoming audio from Exotel — forward to STT with echo suppression."""
        self._audio_chunks_received += 1
        now = asyncio.get_event_loop().time() * 1000  # ms

        # Log first chunk
        if self._audio_chunks_received == 1:
            await self._send_log(f"Receiving mic audio... (STT connected={self.stt._connected if self.stt else 'N/A'})")

        # ECHO SUPPRESSION: Block audio during TTS generation + remaining phone playback.
        # Phone continues playing audio long after TTS finishes generating;
        # blocking only 500ms was insufficient — echo of order items passed through.
        if self.tts and self.tts.is_speaking:
            return
        if self._tts_last_finished > 0:
            elapsed_ms = now - self._tts_last_finished
            playback_buffer_ms = max(self.ECHO_BUFFER_MS, self._remaining_playback_sec * 1000)
            if elapsed_ms < playback_buffer_ms:
                return

        # Don't send if call already ended
        if self._call_ended:
            return

        if self.stt and self.stt._connected:
            sent = await self.stt.send_audio(payload)
            if sent:
                self._audio_chunks_sent_to_stt += 1
                self._last_audio_sent_time = now

                # Schedule a server-side flush after silence
                self._schedule_flush()

        if self._audio_chunks_received % 100 == 0:
            await self._send_log(f"Audio: {self._audio_chunks_received} recv, {self._audio_chunks_sent_to_stt} sent to STT")

    def _schedule_flush(self):
        """Schedule a flush after FLUSH_AFTER_MS of no new audio to STT."""
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        self._flush_task = asyncio.create_task(self._delayed_flush())

    async def _delayed_flush(self):
        """Wait for silence then flush STT."""
        await asyncio.sleep(self.FLUSH_AFTER_MS / 1000)
        # Only flush if no new audio was sent during the wait
        now = asyncio.get_event_loop().time() * 1000
        if (now - self._last_audio_sent_time) >= (self.FLUSH_AFTER_MS - 100):
            if self.stt and self.stt._connected and not self._call_ended:
                await self._send_log("Server-side silence detected — flushing STT")
                await self.stt.flush()

    async def handle_flush(self):
        """Handle flush signal from browser — user stopped speaking."""
        await self._send_log("User stopped speaking — flushing STT...")
        if self.stt:
            await self.stt.flush()

    def _is_echo(self, text: str) -> bool:
        """Detect if a transcript is the agent's own voice echoed back.

        Short responses (< ECHO_MIN_WORDS) are never treated as echo,
        since real user replies like "ஓகே" should always pass through.
        For longer transcripts, check word overlap with the last agent utterance.
        """
        if not self._last_agent_text:
            return False
        words = text.split()
        if len(words) < self.ECHO_MIN_WORDS:
            return False  # Short responses are never echo
        text_words = set(w.lower() for w in words)
        agent_words = set(w.lower() for w in self._last_agent_text.split())
        overlap = len(text_words & agent_words) / len(text_words)
        return overlap >= self.ECHO_WORD_OVERLAP_THRESHOLD

    async def _on_transcript(self, text: str, is_final: bool):
        """Called when STT produces a transcript."""
        text = text.strip()

        # Ignore empty transcripts
        if not text:
            return

        # During greeting intro phase, any vendor response means they're listening
        if self._greeting_phase == 1 and is_final:
            await self._send_log(f"Vendor response during intro: '{text}'")
            self._intro_ack_event.set()
            return

        # Ignore very short transcripts (noise/echo artifacts)
        if len(text) < 3:
            logger.info(f"FILTER: short transcript (<3 chars): '{text}'")
            await self._send_log(f"Ignored short transcript: '{text}'")
            return

        # Ignore transcripts from ultra-short speech (< 200ms = noise)
        if self._last_speech_duration_ms < self.MIN_SPEECH_DURATION_MS:
            logger.info(f"FILTER: noise ({self._last_speech_duration_ms:.0f}ms < {self.MIN_SPEECH_DURATION_MS}ms): '{text}'")
            await self._send_log(f"Ignored noise ({self._last_speech_duration_ms:.0f}ms < {self.MIN_SPEECH_DURATION_MS}ms): '{text}'")
            return

        # During confirmation wait, filter short ambiguous transcripts (noise like "ம்.", "ஆ")
        # But allow them during normal conversation (vendor might say "5", "12" for modify)
        if self._confirmation_pending and len(text) <= 4:
            logger.info(f"FILTER: short during confirmation: '{text}'")
            await self._send_log(f"Ignored short transcript during confirmation wait: '{text}'")
            return

        # Echo detection: if transcript is a long phrase that overlaps heavily with agent's last speech, ignore
        if self._is_echo(text):
            await self._send_log(f"Echo detected, ignoring: '{text}'")
            return

        logger.info(f"ACCEPTED transcript: '{text}' (duration={self._last_speech_duration_ms:.0f}ms, pending={self._confirmation_pending})")
        await self._send_log(f"You said: {text}")

        if not is_final:
            return

        if self._processing:
            # Queue latest transcript instead of dropping — will process after current LLM finishes
            self._queued_transcript = text
            await self._send_log(f"Queued transcript (LLM busy): '{text}'")
            return

        self._processing = True
        self._silence_prompts_sent = 0  # Reset on real user response
        try:
            await self._send_log(f"Thinking...")
            response = await self.llm.chat(text)
            await self._send_log(f"Agent raw: {response}")

            # Parse <speak> and <status> tags from LLM response
            speak_text, status = self._parse_llm_response(response)

            await self._send_log(f"Agent speak: {speak_text} | status: {status}")

            # Track for echo detection
            self._last_agent_text = speak_text

            # Speak only the <speak> content (not status tags)
            if speak_text:
                await self._speak(speak_text)

            # Track UNCLEAR_RESPONSE — only end call after repeated failures
            if "UNCLEAR" in status.upper():
                self._unclear_count += 1
                await self._send_log(f"Unclear response ({self._unclear_count}/{self._max_unclear_before_end})")
                if self._unclear_count >= self._max_unclear_before_end:
                    await self._send_webhook("UNCLEAR_RESPONSE")
                    await self._finish_call("UNCLEAR_RESPONSE")
                    return
                # Otherwise let the agent keep trying — don't end call
                return
            else:
                self._unclear_count = 0  # Reset on any clear response

            terminal = self._extract_terminal_status(status)
            logger.info(f"STATUS: parsed='{status}' terminal={terminal} pending={self._confirmation_pending} is_question={self._speak_is_question(speak_text)}")

            if terminal:
                if self._confirmation_pending == terminal:
                    # User confirmed — agent already asked, now end the call
                    if terminal == "REJECTED":
                        new_reason = self._extract_reason_from_status(status) or text.strip()
                        if new_reason and new_reason not in self._rejection_reason:
                            self._rejection_reason = (self._rejection_reason + " | " + new_reason) if self._rejection_reason else new_reason
                    elif terminal == "MODIFIED":
                        new_reason = self._extract_reason_from_status(status) or text.strip()
                        if new_reason and new_reason not in self._modification_reason:
                            self._modification_reason = (self._modification_reason + " | " + new_reason) if self._modification_reason else new_reason
                    logger.info(f"CALL END: User confirmed {terminal}")
                    await self._send_log(f"User confirmed {terminal} — ending call")
                    await self._send_webhook(terminal)
                    await self._finish_call(terminal)
                    return
                elif self._speak_is_question(speak_text):
                    # Agent is asking a confirmation question — set pending, wait for user
                    self._confirmation_pending = terminal
                    if terminal == "REJECTED":
                        reason = self._extract_reason_from_status(status)
                        if reason:
                            self._rejection_reason = reason
                    elif terminal == "MODIFIED":
                        reason = self._extract_reason_from_status(status)
                        if reason:
                            self._modification_reason = reason
                    logger.info(f"CONFIRMATION SET: pending={terminal}")
                    await self._send_log(f"Confirmation pending for {terminal} — waiting for user YES")
                else:
                    # Terminal status with no question — end call directly
                    logger.info(f"CALL END: Terminal {terminal} with no question — ending directly")
                    await self._send_webhook(terminal)
                    await self._finish_call(terminal)
                    return
            elif not terminal:
                # LLM returned non-terminal (e.g. CONFIRMING) — check if speech implies call is done
                implied = self._speak_implies_call_done(speak_text)
                if implied:
                    logger.info(f"CALL END: LLM said CONFIRMING but speech implies {implied} — ending call")
                    await self._send_webhook(implied)
                    await self._finish_call(implied)
                    return
                elif self._confirmation_pending:
                    logger.info(f"WAITING: Still pending {self._confirmation_pending}, LLM returned non-terminal")
                    await self._send_log(f"Still waiting for {self._confirmation_pending} confirmation")
            # Silence timeout will restart when TTS finishes (in _on_tts_done)
        except Exception as e:
            await self._send_log(f"Error: {e}")
        finally:
            self._processing = False
            # Process queued transcript if one arrived while LLM was busy
            if self._queued_transcript and not self._call_ended:
                queued = self._queued_transcript
                self._queued_transcript = None
                await self._send_log(f"Processing queued transcript: '{queued}'")
                await self._on_transcript(queued, True)

    def _is_user_accepting(self, transcript: str) -> bool:
        """Check if the user's transcript contains clear acceptance words."""
        lower = transcript.lower().strip()
        for kw in self._acceptance_keywords:
            if kw.lower() in lower:
                return True
        return False

    async def _finish_call(self, status: str):
        """End the call after current speech finishes playing."""
        # Wait for TTS to finish generating
        while self.tts and self.tts.is_speaking:
            await asyncio.sleep(0.1)
        await self._end_call(status)

    def _speak_is_question(self, text: str) -> bool:
        """Check if the speak text is a question (agent is asking something and should wait)."""
        if not text:
            return False
        question_markers = ["?", "ஏன்", "இல்லையா", "என்ன", "முடியுமா", "சொல்லுங்க"]
        for marker in question_markers:
            if marker in text:
                return True
        return False

    def _speak_implies_call_done(self, text: str) -> Optional[str]:
        """Detect if the agent's speech implies the call is done (confirmed/rejected/modified).
        Returns the implied terminal status or None.
        LLM sometimes returns CONFIRMING status even when speech says 'confirm done, thanks'.
        """
        if not text:
            return None
        lower = text.lower()
        # Agent says order is confirmed/accepted + thanks/bye
        accept_phrases = ["confirm பண்ணிட்டேன்", "போட்டுட்டேன்", "confirm ஆயிடுச்சு", "accept ஆயிடுச்சு"]
        reject_phrases = ["reject பண்ணிட்டேன்", "noted", "புரிஞ்சது"]
        modify_phrases = ["modify request போட்டுட்டேன்", "forward பண்ணிட்டேன்", "request போட்டுட்டேன்"]

        for phrase in modify_phrases:
            if phrase in lower:
                return "MODIFIED"
        for phrase in reject_phrases:
            if phrase in lower:
                return "REJECTED"
        for phrase in accept_phrases:
            if phrase in lower:
                return "ACCEPTED"
        return None

    def _parse_llm_response(self, response: str) -> tuple[str, str]:
        """Parse <speak> and <status> tags from LLM response.
        Returns (speak_text, status_text). Falls back to full response if no tags.
        """
        # Extract <speak>...</speak>
        speak_match = re.search(r'<speak>(.*?)</speak>', response, re.DOTALL)
        if speak_match:
            speak_text = speak_match.group(1).strip()
        else:
            # Handle missing </speak> — extract everything after <speak>
            open_match = re.search(r'<speak>(.*)', response, re.DOTALL)
            speak_text = open_match.group(1).strip() if open_match else ""

        # Extract <status>...</status>
        status_match = re.search(r'<status>(.*?)</status>', response, re.DOTALL)
        status_text = status_match.group(1).strip() if status_match else ""

        # Fallback: if no <speak> tag found
        if not speak_text:
            if status_text:
                # Has <status> but no <speak> — text before <status> is the speech
                before_status = re.split(r'<status>', response, maxsplit=1)[0].strip()
                speak_text = before_status
            else:
                # No tags at all — treat entire response as speech
                speak_text = response.strip()
                status_text = self._detect_status_fallback(response)

        # Always strip any leftover XML tags from speak text
        speak_text = re.sub(r'</?(?:speak|status)(?:\s[^>]*)?>.*', '', speak_text, flags=re.DOTALL).strip()
        speak_text = re.sub(r'</?(?:speak|status)>', '', speak_text).strip()

        return speak_text, status_text

    def _extract_reason_from_status(self, status: str) -> Optional[str]:
        """Extract rejection reason from status string like 'REJECTED | REASON: ...'"""
        match = re.search(r'REASON:\s*(.+)', status, re.IGNORECASE)
        return match.group(1).strip() if match else None

    def _extract_terminal_status(self, status: str) -> Optional[str]:
        """Check if the status string is a terminal status that ends the call.
        Handles 'REJECTED | REASON: ...' and 'MODIFIED | REASON: ...' formats.
        """
        if not status:
            return None
        upper = status.upper().strip()

        # Handle REJECTED | REASON: ...
        if upper.startswith("REJECTED"):
            reason_match = re.search(r'REASON:\s*(.+)', status, re.IGNORECASE)
            if reason_match:
                self._rejection_reason = reason_match.group(1).strip()
            return "REJECTED"

        # Handle MODIFIED | REASON: ...
        if upper.startswith("MODIFIED"):
            reason_match = re.search(r'REASON:\s*(.+)', status, re.IGNORECASE)
            if reason_match:
                self._modification_reason = reason_match.group(1).strip()
            return "MODIFIED"

        for terminal in self._terminal_statuses:
            if terminal in upper:
                return terminal

        return None

    def _detect_status_fallback(self, response: str) -> str:
        """Fallback: detect status from response text when LLM doesn't use tags."""
        lower = response.lower()

        # Skip double-check questions
        if "பண்ணிட்டுமா" in lower or "பண்றீங்களா" in lower or "ஓகே-வா" in lower:
            return "CONFIRMING"

        upper = response.upper().strip()
        for keyword in self._terminal_statuses:
            if keyword in upper:
                return keyword

        has_thanks = any(s in lower for s in ["தேங்க்ஸ்", "thanks", "நன்றி"])
        if any(s in lower for s in ["accept ஆயிடுச்சு", "அக்செப்ட் ஆயிடுச்சு"]) and has_thanks:
            return "ACCEPTED"
        if any(s in lower for s in ["reject ஆயிடுச்சு", "ரிஜெக்ட் ஆயிடுச்சு"]) and has_thanks:
            return "REJECTED"
        if any(s in lower for s in ["அப்புறம் கால் பண்றேன்", "அப்புறம் ட்ரை பண்றேன்"]):
            return "CALLBACK_REQUESTED"
        if any(s in lower for s in ["கிளியரா சொல்ல முடியல", "கிளியரா புரியல"]):
            return "UNCLEAR_RESPONSE"

        return "CONFIRMING"

    async def _send_webhook(self, status: str):
        """Send order confirmation result to n8n webhook."""
        if self._webhook_sent:
            return
        self._webhook_sent = True

        payload = {
            "order_id": self.order_data["order_id"],
            "vendor_name": self.order_data["vendor_name"],
            "company": self.order_data["company_name"],
            "total_amount": config._calc_total(self.order_data),
            "status": status,
            "call_sid": self.call_sid,
        }
        if status == "REJECTED" and self._rejection_reason:
            payload["rejection_reason"] = self._rejection_reason
        if status == "MODIFIED" and self._modification_reason:
            payload["modification_reason"] = self._modification_reason

        await self._send_log(f"Sending webhook: Status={status}")
        logger.info(f"Sending webhook to {config.WEBHOOK_URL}: {payload}")

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.post(config.WEBHOOK_URL, json=payload)
                await self._send_log(f"Webhook sent: {resp.status_code}")
                logger.info(f"Webhook response: {resp.status_code} {resp.text[:200]}")
        except Exception as e:
            await self._send_log(f"Webhook error: {e}")
            logger.error(f"Webhook error: {e}")

    async def _end_call(self, status: str):
        """Notify the browser of the result and hang up the Exotel call."""
        self._call_ended = True
        status_labels = {
            "ACCEPTED": "Order Accepted",
            "REJECTED": "Order Rejected",
            "MODIFIED": "Order Modification Requested",
            "CALLBACK_REQUESTED": "Callback Requested",
            "NO_RESPONSE": "No Response",
            "UNCLEAR_RESPONSE": "Unclear Response",
        }
        label = status_labels.get(status, status)

        # Wait for TTS to finish playing the goodbye message
        await asyncio.sleep(5)

        # Notify browser tester (if connected via browser)
        try:
            await self.exotel_ws.send_json({
                "event": "end_call",
                "status": status,
                "message": f"Call ended — {label}",
            })
        except Exception:
            pass

        # Hang up the real Exotel call via REST API
        await self._hangup_exotel_call()

        await self._send_log(f"Call ended — {label}")
        logger.info(f"Call ended for {self.call_sid}: {status}")

    async def _hangup_exotel_call(self):
        """Hang up the Exotel call by closing the WebSocket.

        Exotel advances to the next applet (Hangup) when the WebSocket closes.
        """
        if not self.call_sid or self.call_sid.startswith("test-"):
            return  # Skip for browser tester

        try:
            await self.exotel_ws.close()
            await self._send_log("Closed WebSocket — Exotel will hangup")
            logger.info("Closed WebSocket to trigger Exotel hangup")
        except Exception as e:
            logger.error(f"WebSocket close error: {e}")

    async def _on_tts_done(self):
        """Called when TTS finishes generating — NON-BLOCKING.

        Must not sleep here because this runs inside the TTS listener loop.
        Sleeping would block audio chunk processing and cause race conditions.

        Echo suppression strategy:
        - Audio-level: mic is blocked during TTS generation + 500ms buffer (handled in handle_media)
        - Transcript-level: any remaining echo is caught by _is_echo() in _on_transcript
        """
        # Signal browser tester to flush accumulated MP3 chunks
        try:
            await self.exotel_ws.send_json({"event": "tts_done"})
        except Exception:
            pass

        now = asyncio.get_event_loop().time()
        self._tts_last_finished = now * 1000
        self._cancel_silence_timeout()

        is_browser = self.call_sid.startswith("test-")
        if is_browser:
            # Browser tester: audio is MP3 (compressed, ~192kbps ≈ 24KB/s)
            # and is buffered until tts_done flush — playback starts AFTER generation
            playback_sec = self._tts_audio_bytes / 24000 if self._tts_audio_bytes > 0 else 0.0
            self._remaining_playback_sec = playback_sec  # full duration, not reduced
        else:
            # Real telephony: raw PCM streams to phone during generation
            sample_rate = config.TTS_SAMPLE_RATE_TELEPHONY
            playback_sec = self._tts_audio_bytes / (sample_rate * 2) if self._tts_audio_bytes > 0 else 0.0
            generation_sec = now - self._tts_speak_start if self._tts_speak_start > 0 else 0
            self._remaining_playback_sec = max(0, playback_sec - generation_sec)

        if playback_sec < self.MIN_TTS_DURATION_FOR_SILENCE_TIMEOUT:
            await self._send_log(f"Short TTS ({playback_sec:.1f}s) — skipping silence timeout")
            return

        await self._send_log(
            f"TTS done — playback ≈ {playback_sec:.1f}s, remaining ≈ {self._remaining_playback_sec:.1f}s"
        )

        if not self._call_ended and not self._processing:
            self._start_silence_timeout()

    def _start_silence_timeout(self):
        """Start a timeout that re-prompts the vendor if they don't respond."""
        if self._silence_timeout_task and not self._silence_timeout_task.done():
            self._silence_timeout_task.cancel()
        self._silence_timeout_task = asyncio.create_task(self._silence_timeout_handler())

    def _cancel_silence_timeout(self):
        """Cancel the silence timeout (user responded)."""
        if self._silence_timeout_task and not self._silence_timeout_task.done():
            self._silence_timeout_task.cancel()
            self._silence_timeout_task = None

    async def _silence_timeout_handler(self):
        """Wait for remaining playback + silence_timeout_sec, then re-prompt or end call."""
        try:
            # Wait for audio to finish playing on the phone, THEN wait for silence
            total_wait = self._remaining_playback_sec + self._silence_timeout_sec
            await asyncio.sleep(total_wait)

            if self._call_ended or self._processing:
                return

            # If confirmation is pending and vendor stayed silent, end call now
            if self._confirmation_pending:
                await self._send_log(f"Silence after confirmation question — ending with {self._confirmation_pending}")
                goodbye = "சரி... நன்றி."
                self._last_agent_text = goodbye
                await self._speak(goodbye)
                await self._send_webhook(self._confirmation_pending)
                await self._finish_call(self._confirmation_pending)
                return

            self._silence_prompts_sent += 1
            if self._silence_prompts_sent > self._max_silence_prompts:
                # Too many unanswered prompts — end call
                await self._send_log("No response after multiple prompts — ending call")
                response = "சரி... அப்புறம் ட்ரை பண்றேன்."
                self._last_agent_text = response
                await self._speak(response)
                await self._send_webhook("NO_RESPONSE")
                await self._end_call("NO_RESPONSE")
                return

            # Re-prompt the vendor with colloquial Tamil
            await self._send_log(f"Silence timeout — prompting vendor (attempt {self._silence_prompts_sent})")
            prompts = [
                "ஹலோ... ஆர்டர் ஓகே-வா? இருக்கீங்களா?",
                "ஹலோ... இருக்கீங்களா? ஆர்டர் confirm பண்ணலாமா?",
            ]
            prompt = prompts[min(self._silence_prompts_sent - 1, len(prompts) - 1)]
            self._last_agent_text = prompt
            await self._speak(prompt)
            # Next silence timeout will start when TTS finishes (in _on_tts_done)
        except asyncio.CancelledError:
            pass

    async def _on_vad(self, signal: str):
        """Handle VAD events from Saaras v3 STT."""
        if signal == "speech_start":
            # Cancel silence timeout IMMEDIATELY — before any awaits
            self._cancel_silence_timeout()
            self._vad_speech_start_time = asyncio.get_event_loop().time()
            await self._send_log("VAD: user speech detected")
        elif signal == "speech_end":
            now = asyncio.get_event_loop().time()
            self._last_speech_duration_ms = (now - self._vad_speech_start_time) * 1000 if self._vad_speech_start_time > 0 else 999
            await self._send_log(f"VAD: user speech ended ({self._last_speech_duration_ms:.0f}ms) — flushing STT")
            if self.stt and self.stt._connected and not self._call_ended:
                await self.stt.flush()

    async def _speak(self, text: str):
        if self.tts:
            self._cancel_silence_timeout()
            self._tts_audio_bytes = 0
            self._tts_speak_start = asyncio.get_event_loop().time()
            await self.tts.speak(text)

    async def _on_tts_audio(self, audio_base64: str):
        # Track audio bytes for playback duration estimation
        self._tts_audio_bytes += len(audio_base64) * 3 // 4  # base64 → raw bytes
        try:
            await self.exotel_ws.send_json({
                "event": "media",
                "stream_sid": self.stream_sid,
                "media": {
                    "payload": audio_base64,
                }
            })
        except Exception as e:
            logger.error(f"Error sending audio: {e}")

    def _release_key(self):
        """Return the API key to the pool. Guarded against double-release."""
        if self._key_released or not self._on_key_release:
            return
        self._key_released = True
        try:
            self._on_key_release()
        except Exception as e:
            logger.error(f"Error releasing API key: {e}")

    async def _greeting_fallback_timeout(self):
        """Safety net: if _on_tts_done never fires after greeting, start silence timeout."""
        try:
            await asyncio.sleep(15)
            if not self._call_ended and not self._processing and not (
                self._silence_timeout_task and not self._silence_timeout_task.done()
            ):
                await self._send_log("Greeting fallback: TTS done never fired — starting silence timeout")
                self._start_silence_timeout()
        except asyncio.CancelledError:
            pass

    async def stop(self):
        logger.info(f"Agent stopping for call {self.call_sid}")
        self._cancel_silence_timeout()
        if self._greeting_fallback_task and not self._greeting_fallback_task.done():
            self._greeting_fallback_task.cancel()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()
        if self.stt:
            await self.stt.close()
        if self.tts:
            await self.tts.close()
        if self.llm:
            await self.llm.close()
        self._release_key()
        logger.info(f"Agent stopped for call {self.call_sid}")
