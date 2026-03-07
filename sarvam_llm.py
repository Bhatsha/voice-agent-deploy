import logging
import re
from typing import List, Dict, Optional

import httpx

import config

logger = logging.getLogger(__name__)

# Strip <think>...</think> tags from LLM output
_THINK_PATTERN = re.compile(r'<think>.*?</think>', re.DOTALL)


class SarvamLLM:
    """Chat completion client for Sarvam AI with streaming support."""

    def __init__(self, system_prompt: Optional[str] = None):
        self.messages: List[Dict[str, str]] = [
            {"role": "system", "content": system_prompt or "You are a helpful assistant."}
        ]
        self.client = httpx.AsyncClient(timeout=30.0)

    async def chat(self, user_message: str) -> str:
        """Send user message and get assistant response (non-streaming)."""
        self.messages.append({"role": "user", "content": user_message})

        try:
            response = await self.client.post(
                config.OPENAI_LLM_URL,
                headers={
                    "Authorization": f"Bearer {config.OPENAI_API_KEY}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": config.OPENAI_LLM_MODEL,
                    "messages": self.messages,
                    "temperature": 0.3,
                    "max_tokens": 200,
                },
            )
            response.raise_for_status()
            data = response.json()

            assistant_message = data["choices"][0]["message"]["content"]
            # Strip <think>...</think> reasoning tags if present
            assistant_message = _THINK_PATTERN.sub('', assistant_message).strip()
            self.messages.append({"role": "assistant", "content": assistant_message})

            logger.info(f"LLM response: {assistant_message[:80]}...")
            return assistant_message

        except Exception as e:
            logger.error(f"LLM error: {e}")
            return "மன்னிக்கவும், ஒரு தொழில்நுட்ப சிக்கல் ஏற்பட்டுள்ளது."

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()
