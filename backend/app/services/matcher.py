from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Set

from backend.app.schemas.jd_schema import JDSchema, JDRequirement
from backend.app.schemas.score_schema import (
    AnalysisReportSchema,
    MatchEvidence,
    MatchScores,
    RequirementMatchResult,
)
from backend.app.schemas.resume_schema import ResumeBullet, ResumeSchema


def normalize_text(text: str) -> str:
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9+#./\-\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_token(text: str) -> str:
    return normalize_text(text).strip()


def unique_preserve_order(items: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for item in items:
        cleaned = item.strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result


def flatten_resume_bullets(resume: ResumeSchema) -> List[ResumeBullet]:
    bullets: List[ResumeBullet] = []
    for exp in resume.experience:
        bullets.extend(exp.bullets)
    for proj in resume.projects:
        bullets.extend(proj.bullets)
    return bullets


def collect_resume_skill_strings(resume: ResumeSchema) -> List[str]:
    skills = []
    skills.extend(resume.raw_skills)
    skills.extend(resume.inferred_skills)

    for bullet in flatten_resume_bullets(resume):
        skills.extend(bullet.technologies)

    return unique_preserve_order(skills)


def build_resume_skill_set(resume: ResumeSchema) -> Set[str]:
    return {
        normalize_token(skill)
        for skill in collect_resume_skill_strings(resume)
        if normalize_token(skill)
    }


def build_resume_text_corpus(resume: ResumeSchema) -> str:
    parts: List[str] = []

    if resume.summary:
        parts.append(resume.summary)

    parts.extend(resume.raw_skills)
    parts.extend(resume.inferred_skills)

    for bullet in flatten_resume_bullets(resume):
        parts.append(bullet.text)
        parts.extend(bullet.technologies)
        parts.extend(bullet.claims)

    return normalize_text(" \n ".join(parts))


def tokenize_keywords(requirement: JDRequirement) -> List[str]:
    keywords = unique_preserve_order(requirement.keywords + [requirement.text])
    return [normalize_token(k) for k in keywords if normalize_token(k)]


def keyword_in_text(keyword: str, text: str) -> bool:
    """
    Conservative substring/word-ish match.
    """
    if not keyword:
        return False

    # Exact-ish boundary match first
    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    if re.search(pattern, text):
        return True

    # Fallback substring
    return keyword in text


def keyword_in_skill_set(keyword: str, skill_set: Set[str]) -> bool:
    if keyword in skill_set:
        return True

    # Loose containment either way
    for skill in skill_set:
        if keyword in skill or skill in keyword:
            return True
    return False


def requirement_weight(importance: str) -> float:
    if importance == "required":
        return 1.0
    if importance == "preferred":
        return 0.6
    return 0.3


@dataclass
class RequirementEvaluation:
    matched_by_skill: bool
    matched_by_text: bool
    matched_keywords: List[str]
    missing_keywords: List[str]
    evidence_bullets: List[ResumeBullet]


class Matcher:
    def __init__(self) -> None:
        pass

    def match(self, resume: ResumeSchema, jd: JDSchema) -> AnalysisReportSchema:
        resume_skill_strings = collect_resume_skill_strings(resume)
        resume_skill_set = build_resume_skill_set(resume)
        resume_text_corpus = build_resume_text_corpus(resume)
        resume_bullets = flatten_resume_bullets(resume)

        requirement_matches: List[RequirementMatchResult] = []

        matched_skill_names: List[str] = []
        missing_skill_names: List[str] = []
        matched_keywords_global: List[str] = []
        missing_keywords_global: List[str] = []
        strongest_evidence_bullets: List[str] = []

        weighted_requirement_score = 0.0
        total_requirement_weight = 0.0

        weighted_keyword_hits = 0.0
        weighted_keyword_total = 0.0

        responsibility_coverage_hits = 0.0

        for req in jd.requirements:
            evaluation = self._evaluate_requirement(
                req=req,
                resume_skill_set=resume_skill_set,
                resume_text_corpus=resume_text_corpus,
                resume_bullets=resume_bullets,
            )

            weight = requirement_weight(req.importance)
            total_requirement_weight += weight

            matched = evaluation.matched_by_skill or evaluation.matched_by_text
            if matched:
                weighted_requirement_score += weight

            weighted_keyword_hits += weight * len(evaluation.matched_keywords)
            weighted_keyword_total += weight * max(
                len(evaluation.matched_keywords) + len(evaluation.missing_keywords), 1
            )

            evidence = MatchEvidence(
                requirement_id=req.id,
                matched=matched,
                confidence=self._estimate_confidence(evaluation),
                evidence_bullet_ids=[b.id for b in evaluation.evidence_bullets],
                evidence_texts=[b.text for b in evaluation.evidence_bullets],
                rationale=self._build_rationale(req, evaluation),
            )

            requirement_matches.append(
                RequirementMatchResult(
                    requirement_id=req.id,
                    requirement_text=req.text,
                    importance=req.importance,
                    matched_by_skill=evaluation.matched_by_skill,
                    matched_by_semantics=evaluation.matched_by_text,
                    matched_keywords=evaluation.matched_keywords,
                    missing_keywords=evaluation.missing_keywords,
                    evidence=evidence,
                )
            )

            if matched:
                matched_skill_names.append(req.text)
            else:
                missing_skill_names.append(req.text)

            matched_keywords_global.extend(evaluation.matched_keywords)
            missing_keywords_global.extend(evaluation.missing_keywords)
            strongest_evidence_bullets.extend([b.text for b in evaluation.evidence_bullets])

        # Very rough responsibility coverage:
        for resp in jd.responsibilities:
            if self._responsibility_has_evidence(resp.text, resume_bullets, resume_text_corpus):
                responsibility_coverage_hits += 1.0

        skill_coverage_score = (
            weighted_requirement_score / total_requirement_weight
            if total_requirement_weight > 0
            else 0.0
        )

        keyword_alignment_score = (
            weighted_keyword_hits / weighted_keyword_total
            if weighted_keyword_total > 0
            else 0.0
        )

        responsibility_coverage_score = (
            responsibility_coverage_hits / len(jd.responsibilities)
            if jd.responsibilities
            else 0.0
        )

        # Placeholder for future embedding-based score
        semantic_similarity_score = responsibility_coverage_score

        overall_score = (
            0.4 * skill_coverage_score
            + 0.25 * keyword_alignment_score
            + 0.20 * semantic_similarity_score
            + 0.15 * responsibility_coverage_score
        )

        scores = MatchScores(
            skill_coverage_score=round(skill_coverage_score, 4),
            keyword_alignment_score=round(keyword_alignment_score, 4),
            semantic_similarity_score=round(semantic_similarity_score, 4),
            responsibility_coverage_score=round(responsibility_coverage_score, 4),
            overall_score=round(overall_score, 4),
        )

        report = AnalysisReportSchema(
            candidate_name=resume.candidate_name,
            job_title=jd.job_title,
            company_name=jd.company_name,
            matched_skills=unique_preserve_order(matched_skill_names),
            missing_skills=unique_preserve_order(missing_skill_names),
            matched_keywords=unique_preserve_order(matched_keywords_global),
            missing_keywords=unique_preserve_order(missing_keywords_global),
            scores=scores,
            requirement_matches=requirement_matches,
            priority_gaps=self._build_priority_gaps(requirement_matches),
            strongest_evidence_bullets=unique_preserve_order(strongest_evidence_bullets)[:5],
            overall_assessment=self._build_overall_assessment(
                scores=scores,
                matched_skills=unique_preserve_order(matched_skill_names),
                missing_skills=unique_preserve_order(missing_skill_names),
            ),
        )

        return report

    def _evaluate_requirement(
        self,
        req: JDRequirement,
        resume_skill_set: Set[str],
        resume_text_corpus: str,
        resume_bullets: List[ResumeBullet],
    ) -> RequirementEvaluation:
        normalized_req_text = normalize_token(req.text)
        candidate_keywords = tokenize_keywords(req)

        if normalized_req_text and normalized_req_text not in candidate_keywords:
            candidate_keywords = [normalized_req_text] + candidate_keywords

        candidate_keywords = unique_preserve_order(candidate_keywords)

        matched_keywords: List[str] = []
        missing_keywords: List[str] = []

        matched_by_skill = False
        matched_by_text = False

        for kw in candidate_keywords:
            in_skills = keyword_in_skill_set(kw, resume_skill_set)
            in_text = keyword_in_text(kw, resume_text_corpus)

            if in_skills or in_text:
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)

            matched_by_skill = matched_by_skill or in_skills
            matched_by_text = matched_by_text or in_text

        evidence_bullets = self._find_evidence_bullets(candidate_keywords, resume_bullets)

        # If there are good evidence bullets, treat that as text match support
        if evidence_bullets:
            matched_by_text = True

        return RequirementEvaluation(
            matched_by_skill=matched_by_skill,
            matched_by_text=matched_by_text,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            evidence_bullets=evidence_bullets,
        )

    def _find_evidence_bullets(
        self,
        keywords: List[str],
        resume_bullets: List[ResumeBullet],
        top_k: int = 3,
    ) -> List[ResumeBullet]:
        scored = []

        for bullet in resume_bullets:
            bullet_text = normalize_text(
                " ".join(
                    [bullet.text] + bullet.technologies + bullet.claims
                )
            )
            score = 0

            for kw in keywords:
                if keyword_in_text(kw, bullet_text):
                    score += 1

            if score > 0:
                scored.append((score, bullet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [bullet for _, bullet in scored[:top_k]]

    def _responsibility_has_evidence(
        self,
        responsibility_text: str,
        resume_bullets: List[ResumeBullet],
        resume_text_corpus: str,
    ) -> bool:
        normalized_resp = normalize_token(responsibility_text)
        if not normalized_resp:
            return False

        if keyword_in_text(normalized_resp, resume_text_corpus):
            return True

        # Weak heuristic: enough token overlap with any bullet
        resp_tokens = {
            tok for tok in normalized_resp.split()
            if len(tok) > 3
        }

        for bullet in resume_bullets:
            bullet_text = normalize_text(
                " ".join(
                    [bullet.text] + bullet.technologies + bullet.claims
                )
            )
            bullet_tokens = set(bullet_text.split())
            overlap = len(resp_tokens & bullet_tokens)
            if overlap >= 2:
                return True

        return False

    def _estimate_confidence(self, evaluation: RequirementEvaluation) -> float:
        score = 0.0
        if evaluation.matched_by_skill:
            score += 0.45
        if evaluation.matched_by_text:
            score += 0.35
        score += min(len(evaluation.matched_keywords) * 0.08, 0.2)
        return round(min(score, 1.0), 3)

    def _build_rationale(
        self,
        req: JDRequirement,
        evaluation: RequirementEvaluation,
    ) -> str:
        if evaluation.evidence_bullets:
            return (
                f"Requirement '{req.text}' is supported by matching resume content "
                f"and {len(evaluation.evidence_bullets)} relevant bullet(s)."
            )

        if evaluation.matched_by_skill:
            return (
                f"Requirement '{req.text}' appears supported by the extracted resume skill set."
            )

        return f"Requirement '{req.text}' has weak or missing direct evidence in the resume."

    def _build_priority_gaps(
        self,
        requirement_matches: List[RequirementMatchResult],
    ) -> List[str]:
        gaps = []
        for item in requirement_matches:
            if item.importance == "required" and not (
                item.matched_by_skill or item.matched_by_semantics
            ):
                gaps.append(item.requirement_text)

        if not gaps:
            for item in requirement_matches:
                if item.importance in {"required", "preferred"} and item.missing_keywords:
                    gaps.append(item.requirement_text)

        return unique_preserve_order(gaps)[:5]

    def _build_overall_assessment(
        self,
        scores: MatchScores,
        matched_skills: List[str],
        missing_skills: List[str],
    ) -> str:
        if scores.overall_score >= 0.75:
            level = "strong"
        elif scores.overall_score >= 0.5:
            level = "moderate"
        else:
            level = "limited"

        matched_preview = ", ".join(matched_skills[:5]) if matched_skills else "no strong requirement matches"
        missing_preview = ", ".join(missing_skills[:5]) if missing_skills else "no major missing requirements identified"

        return (
            f"Overall alignment appears {level}. "
            f"Strongest overlap: {matched_preview}. "
            f"Primary gaps: {missing_preview}."
        )