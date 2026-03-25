from __future__ import annotations

import re
from pathlib import Path

from backend.app.schemas.jd_schema import JDSchema
from backend.app.services.llm_client import LLMClient


class JDParserError(RuntimeError):
    pass


def load_prompt_template(prompt_path: str | Path) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise JDParserError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def preprocess_jd_text(text: str) -> str:
    cleaned = text.replace("\xa0", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def postprocess_jd(parsed: JDSchema) -> JDSchema:
    parsed.keywords = sorted(set(x.strip() for x in parsed.keywords if x.strip()))

    for req in parsed.requirements:
        req.keywords = sorted(set(x.strip() for x in req.keywords if x.strip()))

    for resp in parsed.responsibilities:
        resp.keywords = sorted(set(x.strip() for x in resp.keywords if x.strip()))

    return parsed


class JDParser:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path = "backend/app/prompts/jd_to_json.txt",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = load_prompt_template(prompt_path)

    def build_prompt(self, jd_text: str) -> str:
        return self.prompt_template.replace("{{jd_text}}", jd_text)

    def parse_text(self, jd_text: str) -> JDSchema:
        cleaned_text = preprocess_jd_text(jd_text)
        prompt = self.build_prompt(cleaned_text)

        try:
            parsed = self.llm_client.complete_json(prompt, JDSchema)
            return postprocess_jd(parsed)
        except Exception as exc:
            raise JDParserError(f"JD parsing failed: {exc}") from exc