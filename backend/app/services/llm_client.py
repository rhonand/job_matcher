from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Optional, Type, TypeVar

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel, ValidationError

load_dotenv()

T = TypeVar("T", bound=BaseModel)


class LLMClientError(RuntimeError):
    pass


@dataclass
class LLMConfig:
    provider: str = os.getenv("LLM_PROVIDER", "openrouter")
    model: Optional[str] = None
    temperature: float = 0.0
    max_retries: int = 2


class LLMClient:
    def __init__(self, config: Optional[LLMConfig] = None) -> None:
        self.config = config or LLMConfig()
        self.provider = self.config.provider.lower()

        if self.provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise LLMClientError("OPENAI_API_KEY is not set.")
            model = self.config.model or os.getenv("OPENAI_MODEL", "gpt-4.1-mini")
            self.model = model
            self.client = OpenAI(api_key=api_key)

        elif self.provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY")
            if not api_key:
                raise LLMClientError("OPENROUTER_API_KEY is not set.")
            model = self.config.model or os.getenv(
                "OPENROUTER_MODEL",
                "openrouter/free",
            )
            self.model = model
            self.client = OpenAI(
                api_key=api_key,
                base_url="https://openrouter.ai/api/v1",
            )

        else:
            raise LLMClientError(f"Unsupported provider: {self.provider}")

    def complete_text(self, prompt: str) -> str:
        try:
            print(f"[LLM] provider={self.provider}, model={self.model}")
            print("[LLM] sending request...")

            response = self.client.chat.completions.create(
                model=self.model,
                temperature=self.config.temperature,
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
            )
            content = response.choices[0].message.content
            if not content:
                raise LLMClientError("Model returned empty content.")
            return content
        except Exception as exc:
            raise LLMClientError(f"LLM text completion failed: {exc}") from exc

    def complete_json(self, prompt: str, schema_model: Type[T]) -> T:
        last_error: Optional[Exception] = None

        for attempt in range(self.config.max_retries + 1):
            try:
                raw_text = self.complete_text(prompt)
                parsed_obj = self._extract_json(raw_text)
                return schema_model.model_validate(parsed_obj)
            except (json.JSONDecodeError, ValidationError, LLMClientError) as exc:
                last_error = exc
                if attempt == self.config.max_retries:
                    break

        raise LLMClientError(f"Failed to obtain valid structured output: {last_error}")

    @staticmethod
    def _extract_json(text: str) -> Any:
        stripped = text.strip()

        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                stripped = "\n".join(lines[1:-1]).strip()

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            stripped = stripped[start : end + 1]

        return json.loads(stripped)