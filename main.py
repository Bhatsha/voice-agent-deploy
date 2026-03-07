import json
import logging
import re
from typing import List, Optional

import httpx
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

try:
    from google.cloud import firestore
    db = firestore.Client()
except Exception:
    firestore = None
    db = None

import config
from agent import VoiceAgent
from sarvam_key_pool import SarvamKeyPool

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(title="Voice Agent - Order Confirmation")

# Active call sessions
sessions: dict[str, VoiceAgent] = {}

# Sarvam API key pool for concurrent call support
key_pool: Optional[SarvamKeyPool] = None

# Firestore client for shared pending orders across container instances (Cloud Run)
PENDING_ORDERS_COLLECTION = "pending_orders"

# In-memory fallback for single-server deployments (no Firestore needed)
pending_orders: dict[str, dict] = {}


@app.on_event("startup")
async def startup():
    global key_pool
    if config.SARVAM_API_KEYS:
        key_pool = SarvamKeyPool(config.SARVAM_API_KEYS)
        logger.info(f"Key pool ready: {key_pool.status()}")
    else:
        logger.warning("No SARVAM_API_KEYS configured — calls will fail")


def _normalize_phone(phone: str) -> str:
    """Normalize phone number to last 10 digits for matching."""
    digits = re.sub(r'\D', '', phone)
    return digits[-10:] if len(digits) >= 10 else digits


@app.get("/")
async def health():
    result = {"status": "ok", "active_calls": len(sessions)}
    if key_pool:
        result["key_pool"] = key_pool.status()
    return result


class OrderItem(BaseModel):
    name: str
    qty: int
    price: float
    variation: Optional[str] = None


class CallRequest(BaseModel):
    phone_number: str
    vendor_name: str
    company_name: str = "Keeggi"
    order_id: str
    items: List[OrderItem]


@app.post("/call")
async def trigger_call(req: CallRequest):
    """Trigger an outbound call to a vendor with dynamic order data."""
    phone = req.phone_number
    logger.info(f"Triggering call to {phone} | order={req.order_id} vendor={req.vendor_name}")

    # Reject early if all keys are in use and queue is full
    if key_pool:
        pool_st = key_pool.status()
        if pool_st["available"] == 0 and pool_st["waiting"] >= key_pool.MAX_QUEUE_WAIT:
            logger.warning(f"Rejecting call — key pool exhausted: {pool_st}")
            return {
                "status": "error",
                "detail": "All lines busy. Try again shortly.",
                "key_pool": pool_st,
            }

    # Build order data dict
    order_data = {
        "order_id": req.order_id,
        "vendor_name": req.vendor_name,
        "company_name": req.company_name,
        "items": [{"name": i.name, "qty": i.qty, "price": i.price, "variation": i.variation} for i in req.items],
    }

    # Store order for WebSocket pickup: Firestore (Cloud Run) or in-memory (single server)
    norm_phone = _normalize_phone(phone)
    if db:
        db.collection(PENDING_ORDERS_COLLECTION).document(norm_phone).set(order_data)
        logger.info(f"Stored pending order in Firestore for {norm_phone}: {req.order_id}")
    else:
        pending_orders[norm_phone] = order_data
        logger.info(f"Stored pending order in memory for {norm_phone}: {req.order_id}")

    # Ensure phone has country code for Exotel
    exotel_phone = phone if phone.startswith("91") else f"91{phone}"

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.post(
                config.EXOTEL_API_URL,
                auth=(config.EXOTEL_API_KEY, config.EXOTEL_API_TOKEN),
                data={
                    "From": exotel_phone,
                    "To": exotel_phone,
                    "CallerId": config.EXOTEL_PHONE_NUMBER,
                    "Url": f"http://my.exotel.com/exoml/start/{config.EXOTEL_APP_ID}",
                },
            )
            logger.info(f"Exotel response: {resp.status_code} {resp.text[:300]}")

            if resp.status_code == 200:
                call_data = resp.json()
                call_sid = call_data.get("Call", {}).get("Sid", "")
                return {
                    "status": "ok",
                    "message": f"Call initiated to {phone}",
                    "call_sid": call_sid,
                    "order_id": req.order_id,
                }
            else:
                # Clean up pending order on failure
                if db:
                    db.collection(PENDING_ORDERS_COLLECTION).document(norm_phone).delete()
                pending_orders.pop(norm_phone, None)
                return {"status": "error", "code": resp.status_code, "detail": resp.text[:300]}
    except Exception as e:
        if db:
            db.collection(PENDING_ORDERS_COLLECTION).document(norm_phone).delete()
        pending_orders.pop(norm_phone, None)
        logger.error(f"Failed to trigger call: {e}")
        return {"status": "error", "detail": str(e)}


@app.websocket("/ws")
async def exotel_websocket(ws: WebSocket):
    """Handle Exotel Voicebot WebSocket connection."""
    await ws.accept()
    logger.info("Exotel WebSocket connected")

    agent = None
    stream_sid = None
    api_key = None  # Checked-out key for this call

    def release_fn():
        """Release the API key back to the pool (called by agent or finally block)."""
        nonlocal api_key
        if api_key and key_pool:
            key_pool.release(api_key)
            api_key = None

    try:
        while True:
            raw = await ws.receive_text()
            data = json.loads(raw)
            event = data.get("event", "")

            if event == "connected":
                logger.info("Exotel stream: connected")

            elif event == "start":
                start_data = data.get("start", {})
                stream_sid = data.get("stream_sid") or start_data.get("stream_sid", "")
                call_sid = start_data.get("call_sid", "")
                from_number = start_data.get("from", "")
                to_number = start_data.get("to", "")
                media_format = start_data.get("media_format", {})

                logger.info(
                    f"Call started: {call_sid} | from={from_number} to={to_number} "
                    f"| format={media_format} | stream={stream_sid}"
                )

                # Checkout an API key from the pool
                if key_pool:
                    try:
                        api_key = await key_pool.checkout(timeout=30)
                    except (TimeoutError, RuntimeError) as e:
                        logger.error(f"Key pool checkout failed for {call_sid}: {e}")
                        break

                # Look up order data: Firestore (Cloud Run) or in-memory (single server)
                order_data = None
                if from_number:
                    norm = _normalize_phone(from_number)
                    if db:
                        doc_ref = db.collection(PENDING_ORDERS_COLLECTION).document(norm)
                        doc = doc_ref.get()
                        if doc.exists:
                            order_data = doc.to_dict()
                            doc_ref.delete()
                            logger.info(f"Found pending order in Firestore for {norm}: {order_data['order_id']}")
                    if not order_data and norm in pending_orders:
                        order_data = pending_orders.pop(norm)
                        logger.info(f"Found pending order in memory for {norm}: {order_data['order_id']}")
                    if not order_data:
                        logger.info(f"No pending order for {norm}, using default")

                # Browser tester sends order data directly in start event
                if not order_data and start_data.get("order_id"):
                    order_data = {
                        "order_id": start_data.get("order_id", ""),
                        "vendor_name": start_data.get("vendor_name", ""),
                        "company_name": start_data.get("company_name", "Keeggi"),
                        "items": start_data.get("items", []),
                    }
                    logger.info(f"Using browser-provided order: {order_data['order_id']}")

                agent = VoiceAgent(
                    exotel_ws=ws,
                    stream_sid=stream_sid,
                    call_sid=call_sid,
                    order_data=order_data,
                    api_key=api_key,
                    on_key_release=release_fn,
                )
                sessions[stream_sid] = agent
                await agent.start()

            elif event == "media":
                if agent:
                    payload = data.get("media", {}).get("payload", "")
                    if payload:
                        await agent.handle_media(payload)

            elif event == "dtmf":
                if agent:
                    digit = data.get("dtmf", {}).get("digit", "")
                    await agent.handle_dtmf(digit)

            elif event == "flush":
                if agent:
                    await agent.handle_flush()

            elif event == "mark":
                mark_name = data.get("mark", {}).get("name", "")
                logger.info(f"Mark received: {mark_name}")

            elif event == "stop":
                reason = data.get("stop", {}).get("reason", "unknown")
                logger.info(f"Call stopped: reason={reason}")
                break

    except WebSocketDisconnect:
        logger.info("Exotel WebSocket disconnected")
    except Exception as e:
        logger.error(f"WebSocket error: {e}", exc_info=True)
    finally:
        if agent:
            await agent.stop()
        # Safety-net release if agent didn't release (e.g., agent never created)
        release_fn()
        if stream_sid and stream_sid in sessions:
            del sessions[stream_sid]
        logger.info("Session cleaned up")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
