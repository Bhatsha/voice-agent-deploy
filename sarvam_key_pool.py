import asyncio
import logging
from typing import Optional

logger = logging.getLogger(__name__)


class SarvamKeyPool:
    """Pool of Sarvam API keys for concurrent call support.

    Each call needs its own API key (STT + TTS WebSockets share one key).
    Keys are checked out FIFO and returned when the call ends.
    """

    MAX_QUEUE_WAIT = 10  # max callers waiting for a key

    def __init__(self, api_keys: list[str]):
        if not api_keys:
            raise ValueError("At least one Sarvam API key is required")
        self._total = len(api_keys)
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        for key in api_keys:
            self._queue.put_nowait(key)
        self._in_use = 0
        self._waiting = 0
        logger.info(f"SarvamKeyPool initialized with {self._total} key(s)")

    async def checkout(self, timeout: float = 30.0) -> str:
        """Get an available API key, waiting up to timeout seconds.

        Raises TimeoutError if no key becomes available.
        Raises RuntimeError if too many callers are already waiting.
        """
        if self._waiting >= self.MAX_QUEUE_WAIT:
            raise RuntimeError(
                f"Key pool queue full ({self._waiting} waiting). "
                "Reject call to avoid unbounded wait."
            )
        self._waiting += 1
        try:
            key = await asyncio.wait_for(self._queue.get(), timeout=timeout)
            self._in_use += 1
            logger.info(
                f"Key checked out (available={self._queue.qsize()}, "
                f"in_use={self._in_use})"
            )
            return key
        except asyncio.TimeoutError:
            raise TimeoutError(
                f"No API key available after {timeout}s "
                f"(total={self._total}, in_use={self._in_use})"
            )
        finally:
            self._waiting -= 1

    def release(self, key: str) -> None:
        """Return an API key to the pool. Safe to call from sync/finally blocks."""
        try:
            self._queue.put_nowait(key)
            self._in_use = max(0, self._in_use - 1)
            logger.info(
                f"Key released (available={self._queue.qsize()}, "
                f"in_use={self._in_use})"
            )
        except asyncio.QueueFull:
            logger.error("Key pool release failed — queue unexpectedly full")

    def status(self) -> dict:
        """Pool metrics for the health endpoint."""
        return {
            "total_keys": self._total,
            "available": self._queue.qsize(),
            "in_use": self._in_use,
            "waiting": self._waiting,
        }
