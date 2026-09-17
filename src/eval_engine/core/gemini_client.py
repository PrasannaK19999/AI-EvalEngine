"""Thin Gemini client: send a prompt, get text back. Nothing metric-specific here"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import types


class GeminiJudgeClient:
    """Wraps the Gemini API for use as an evaluation judge."""

    def __init__(self, model: str = "gemini-3.5-flash-lite") -> None:
        load_dotenv()
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise RuntimeError("GEMINI_API_KEY not found in environment or .env file.")
        self._client = genai.Client(api_key=api_key)
        self._model = model

    def complete(self, prompt: str) -> str:
        """Send a prompt to the model and return its text response."""
        response = self._client.models.generate_content(
            model=self._model,
            contents=prompt,
            config=types.GenerateContentConfig(temperature=0.0),
        )

        # Print judge token usage (the cost of running the evaluation itself).
        usage = response.usage_metadata
        if usage is not None:
            print(
                f"[judge tokens] prompt={usage.prompt_token_count} "
                f"output={usage.candidates_token_count} "
                f"total={usage.total_token_count}"
            )

        text = response.text
        if text is None:
            raise RuntimeError("Gemini returned no text in the response.")
        return text