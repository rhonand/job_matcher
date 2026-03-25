from __future__ import annotations

import json
from pathlib import Path

from backend.app.services.jd_parser import JDParser
from backend.app.services.llm_client import LLMClient, LLMConfig
from backend.app.services.pdf_parser import extract_text_from_pdf
from backend.app.services.resume_parser import ResumeParser
from backend.app.services.matcher import Matcher

def main() -> None:
    root = Path(__file__).resolve().parents[1]

    resume_pdf = root / "sample_data" / "resume.pdf"
    jd_txt = root / "sample_data" / "jd.txt"

    if not resume_pdf.exists():
        raise FileNotFoundError(f"Missing sample resume: {resume_pdf}")
    if not jd_txt.exists():
        raise FileNotFoundError(f"Missing sample JD: {jd_txt}")

    resume_text = extract_text_from_pdf(resume_pdf)
    jd_text = jd_txt.read_text(encoding="utf-8")

    llm_client = LLMClient(
        LLMConfig(
            provider="openrouter",
            model="openrouter/free",
            temperature=0.0,
            max_retries=2,
        )
    )

    resume_parser = ResumeParser(llm_client=llm_client)
    jd_parser = JDParser(llm_client=llm_client)
    matcher = Matcher()

    parsed_resume = resume_parser.parse_text(resume_text)
    parsed_jd = jd_parser.parse_text(jd_text)
    report = matcher.match(parsed_resume, parsed_jd)

    output_dir = root / "output"
    output_dir.mkdir(parents=True, exist_ok=True)

    (output_dir / "parsed_resume.json").write_text(
        parsed_resume.model_dump_json(indent=2),
        encoding="utf-8",
    )

    (output_dir / "parsed_jd.json").write_text(
        parsed_jd.model_dump_json(indent=2),
        encoding="utf-8",
    )

    (output_dir / "match_report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print("Parsed resume and JD successfully.")
    print("Overall score:", report.scores.overall_score)
    print("Matched skills:", report.matched_skills)
    print("Missing skills:", report.missing_skills)
    print("Priority gaps:", report.priority_gaps)

if __name__ == "__main__":
    main()