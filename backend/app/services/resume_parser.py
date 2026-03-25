from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from backend.app.schemas.resume_schema import ResumeSchema
from backend.app.services.llm_client import LLMClient


class ResumeParserError(RuntimeError):
    pass


COMMON_RESUME_SECTIONS = [
    "summary",
    "professional summary",
    "skills",
    "technical skills",
    "experience",
    "work experience",
    "professional experience",
    "projects",
    "education",
    "certifications",
    "languages",
]


def load_prompt_template(prompt_path: str | Path) -> str:
    path = Path(prompt_path)
    if not path.exists():
        raise ResumeParserError(f"Prompt file not found: {path}")
    return path.read_text(encoding="utf-8")


def preprocess_resume_text(text: str) -> str:
    """
    Very light cleanup to make LLM parsing more stable.
    """
    cleaned = text.replace("\xa0", " ")
    cleaned = re.sub(r"\r\n?", "\n", cleaned)
    cleaned = re.sub(r"[ \t]+", " ", cleaned)
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    # Optional: add light markers around common sections if they appear standalone
    lines = []
    for raw_line in cleaned.splitlines():
        line = raw_line.strip()
        low = line.lower()
        if low in COMMON_RESUME_SECTIONS:
            lines.append(f"\n## {line}\n")
        else:
            lines.append(raw_line)

    return "\n".join(lines).strip()


def postprocess_resume(parsed: ResumeSchema) -> ResumeSchema:
    """
    Optional cleanup after Pydantic validation.
    """
    parsed.raw_skills = sorted(set(skill.strip() for skill in parsed.raw_skills if skill.strip()))
    parsed.inferred_skills = sorted(
        set(skill.strip() for skill in parsed.inferred_skills if skill.strip())
    )

    for exp in parsed.experience:
        for bullet in exp.bullets:
            bullet.technologies = sorted(
                set(x.strip() for x in bullet.technologies if x.strip())
            )
            bullet.claims = [x.strip() for x in bullet.claims if x.strip()]

    for proj in parsed.projects:
        for bullet in proj.bullets:
            bullet.technologies = sorted(
                set(x.strip() for x in bullet.technologies if x.strip())
            )
            bullet.claims = [x.strip() for x in bullet.claims if x.strip()]

    return parsed


class ResumeParser:
    def __init__(
        self,
        llm_client: LLMClient,
        prompt_path: str | Path = "backend/app/prompts/resume_to_json.txt",
    ) -> None:
        self.llm_client = llm_client
        self.prompt_template = load_prompt_template(prompt_path)

    def build_prompt(self, resume_text: str) -> str:
        return self.prompt_template.replace("{{resume_text}}", resume_text)

    def parse_text(self, resume_text: str) -> ResumeSchema:
        cleaned_text = preprocess_resume_text(resume_text)
        prompt = self.build_prompt(cleaned_text)

        try:
            parsed = self.llm_client.complete_json(prompt, ResumeSchema)
            return postprocess_resume(parsed)
        except Exception as exc:
            raise ResumeParserError(f"Resume parsing failed: {exc}") from exc