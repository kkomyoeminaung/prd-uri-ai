"""
Google Gemini API Service — Free tier
Model: gemini-1.5-flash (free, fast)
"""

import httpx
import json
import logging
from typing import List, Dict, AsyncGenerator

logger = logging.getLogger(__name__)

GEMINI_BASE = "https://generativelanguage.googleapis.com/v1beta/models"
GEMINI_MODEL = "gemini-1.5-flash"


class GeminiService:
    """
    Google Gemini API with PRD causal system prompt injection.
    Free tier: 15 requests/min, 1500 requests/day.
    """

    def __init__(self, api_key: str = ""):
        self.api_key = api_key
        self.model   = GEMINI_MODEL

    def _url(self, method: str) -> str:
        return f"{GEMINI_BASE}/{self.model}:{method}?key={self.api_key}"

    def _build_body(
        self,
        user_message: str,
        system_prompt: str,
        history: List[Dict] = None,
    ) -> dict:
        """Build Gemini request body with history + system prompt."""
        contents = []

        # Gemini system instruction goes separately
        # History conversion: openai-style → gemini-style
        if history:
            for m in history:
                role = "user" if m["role"] == "user" else "model"
                contents.append({
                    "role": role,
                    "parts": [{"text": m["content"]}]
                })

        # Current user message
        contents.append({
            "role": "user",
            "parts": [{"text": user_message}]
        })

        return {
            "system_instruction": {
                "parts": [{"text": system_prompt}]
            },
            "contents": contents,
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 1024,
            },
        }

    async def chat(
        self,
        user_message: str,
        system_prompt: str,
        history: List[Dict] = None,
    ) -> str:
        """Send message, return full response string."""
        if not self.api_key:
            return "⚠️ GEMINI_API_KEY not set. Add it to Railway environment variables."

        body = self._build_body(user_message, system_prompt, history)

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                r = await client.post(self._url("generateContent"), json=body)
                r.raise_for_status()
                data = r.json()
                return data["candidates"][0]["content"]["parts"][0]["text"]

        except httpx.HTTPStatusError as e:
            if e.response.status_code == 429:
                return "⚠️ Rate limit reached (15 req/min). Please wait a moment."
            elif e.response.status_code == 400:
                return "⚠️ Invalid API key. Check GEMINI_API_KEY in Railway settings."
            logger.error(f"Gemini HTTP error: {e}")
            return f"⚠️ API error {e.response.status_code}"
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return f"⚠️ Error: {str(e)}"

    async def chat_stream(
        self,
        user_message: str,
        system_prompt: str,
        history: List[Dict] = None,
    ) -> AsyncGenerator[str, None]:
        """Streaming version using Gemini streamGenerateContent."""
        if not self.api_key:
            yield "⚠️ GEMINI_API_KEY not set."
            return

        body = self._build_body(user_message, system_prompt, history)

        try:
            async with httpx.AsyncClient(timeout=60) as client:
                async with client.stream(
                    "POST", self._url("streamGenerateContent"), json=body
                ) as resp:
                    resp.raise_for_status()
                    buffer = ""
                    async for chunk in resp.aiter_text():
                        buffer += chunk
                        # Parse complete JSON objects from stream
                        while True:
                            try:
                                # Find complete JSON object
                                start = buffer.find("{")
                                if start == -1:
                                    break
                                # Try to parse
                                obj, end = json.JSONDecoder().raw_decode(buffer, start)
                                buffer = buffer[end:]
                                text = (obj.get("candidates", [{}])[0]
                                           .get("content", {})
                                           .get("parts", [{}])[0]
                                           .get("text", ""))
                                if text:
                                    yield text
                            except json.JSONDecodeError:
                                break
        except Exception as e:
            logger.error(f"Gemini stream error: {e}")
            yield f"⚠️ Stream error: {e}"

    async def is_available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=5) as client:
                r = await client.get(
                    f"{GEMINI_BASE}?key={self.api_key}"
                )
                return r.status_code == 200
        except Exception:
            return False
