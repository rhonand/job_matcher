from __future__ import annotations

import json
from pathlib import Path

from backend.app.schemas.jd_schema import JDSchema
from backend.app.schemas.resume_schema import ResumeSchema
from backend.app.services.embedding_matcher import EmbeddingMatcher
from backend.app.services.matcher import Matcher


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    output_dir = root / "output"

    parsed_resume_path = output_dir / "parsed_resume.json"
    parsed_jd_path = output_dir / "parsed_jd.json"

    if not parsed_resume_path.exists():
        raise FileNotFoundError(f"Missing parsed resume JSON: {parsed_resume_path}")
    if not parsed_jd_path.exists():
        raise FileNotFoundError(f"Missing parsed JD JSON: {parsed_jd_path}")

    parsed_resume = ResumeSchema.model_validate_json(
        parsed_resume_path.read_text(encoding="utf-8")
    )
    parsed_jd = JDSchema.model_validate_json(
        parsed_jd_path.read_text(encoding="utf-8")
    )

    print("Loading embedding model...")
    embedding_matcher = EmbeddingMatcher()

    print("Running hybrid matcher...")
    matcher = Matcher(
        embedding_matcher=embedding_matcher,
        semantic_threshold=0.45,
        semantic_evidence_threshold=0.35,
    )
    report = matcher.match(parsed_resume, parsed_jd)

    (output_dir / "match_report.json").write_text(
        report.model_dump_json(indent=2),
        encoding="utf-8",
    )

    print("Hybrid matching done.")
    print("ATS-like score:", report.scores.ats_like_score)
    print("Semantic score:", report.scores.semantic_score)
    print("Blended score:", report.scores.blended_score)
    print("Semantic gap:", report.scores.semantic_gap)
    print("Priority gaps:", report.priority_gaps)

    print("\nFirst 3 requirement results:")
    for item in report.requirement_matches[:3]:
        print("-" * 60)
        print("Requirement:", item.requirement_text)
        print("ATS matched:", item.ats_matched)
        print("Semantic matched:", item.matched_by_semantics)
        print("Semantic score:", item.semantic_score)
        print("Matched keywords:", item.matched_keywords)
        if item.evidence:
            print("Evidence bullet IDs:", item.evidence.evidence_bullet_ids)


if __name__ == "__main__":
    main()