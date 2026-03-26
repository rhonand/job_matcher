from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Set

from backend.app.schemas.jd_schema import JDSchema, JDRequirement
from backend.app.schemas.score_schema import (
    AnalysisReportSchema,
    MatchEvidence,
    MatchScores,
    RequirementMatchResult,
)
from backend.app.schemas.resume_schema import ResumeBullet, ResumeSchema
from backend.app.services.embedding_matcher import EmbeddingMatcher


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

    for edu in resume.education:
        for part in [edu.degree, edu.field_of_study, edu.institution]:
            if part:
                parts.append(part)

    return normalize_text(" \n ".join(parts))


def tokenize_requirement(req: JDRequirement) -> List[str]:
    parts = [req.text] + req.keywords
    return unique_preserve_order(
        [normalize_token(p) for p in parts if normalize_token(p)]
    )


def keyword_in_text(keyword: str, text: str) -> bool:
    if not keyword:
        return False

    pattern = rf"(?<!\w){re.escape(keyword)}(?!\w)"
    if re.search(pattern, text):
        return True

    return keyword in text


def keyword_in_skill_set(keyword: str, skill_set: Set[str]) -> bool:
    if keyword in skill_set:
        return True

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
    matched_by_keyword: bool
    ats_matched: bool

    matched_by_semantics: bool
    semantic_score: float

    matched_keywords: List[str]
    missing_keywords: List[str]
    evidence_bullets: List[ResumeBullet]


class Matcher:
    def __init__(
        self,
        embedding_matcher: Optional[EmbeddingMatcher] = None,
        semantic_threshold: float = 0.45,
        semantic_evidence_threshold: float = 0.35,
    ) -> None:
        self.embedding_matcher = embedding_matcher
        self.semantic_threshold = semantic_threshold
        self.semantic_evidence_threshold = semantic_evidence_threshold

    def match(self, resume: ResumeSchema, jd: JDSchema) -> AnalysisReportSchema:
        resume_skill_set = build_resume_skill_set(resume)
        resume_text_corpus = build_resume_text_corpus(resume)
        resume_bullets = flatten_resume_bullets(resume)

        requirement_matches: List[RequirementMatchResult] = []

        matched_skills: List[str] = []
        missing_skills: List[str] = []
        matched_keywords_global: List[str] = []
        missing_keywords_global: List[str] = []
        strongest_evidence_bullets: List[str] = []

        weighted_skill_hits = 0.0
        weighted_keyword_hits = 0.0
        weighted_keyword_total = 0.0
        weighted_requirement_total = 0.0

        weighted_requirement_semantic_sum = 0.0
        weighted_responsibility_semantic_sum = 0.0
        weighted_responsibility_keyword_hits = 0.0

        for req in jd.requirements:
            evaluation = self._evaluate_requirement(
                req=req,
                resume=resume,
                resume_skill_set=resume_skill_set,
                resume_text_corpus=resume_text_corpus,
                resume_bullets=resume_bullets,
            )

            weight = requirement_weight(req.importance)
            weighted_requirement_total += weight

            if evaluation.matched_by_skill:
                weighted_skill_hits += weight

            weighted_keyword_hits += weight * len(evaluation.matched_keywords)
            weighted_keyword_total += weight * max(
                len(evaluation.matched_keywords) + len(evaluation.missing_keywords),
                1,
            )

            weighted_requirement_semantic_sum += weight * evaluation.semantic_score

            evidence = MatchEvidence(
                requirement_id=req.id,
                matched=evaluation.ats_matched or evaluation.matched_by_semantics,
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
                    matched_by_keyword=evaluation.matched_by_keyword,
                    ats_matched=evaluation.ats_matched,
                    matched_by_semantics=evaluation.matched_by_semantics,
                    semantic_score=round(evaluation.semantic_score, 4),
                    matched_keywords=evaluation.matched_keywords,
                    missing_keywords=evaluation.missing_keywords,
                    evidence=evidence,
                )
            )

            if evaluation.ats_matched:
                matched_skills.append(req.text)
            else:
                missing_skills.append(req.text)

            matched_keywords_global.extend(evaluation.matched_keywords)
            missing_keywords_global.extend(evaluation.missing_keywords)
            strongest_evidence_bullets.extend([b.text for b in evaluation.evidence_bullets])

        # Responsibility scores
        for resp in jd.responsibilities:
            kw_hit = self._responsibility_keyword_match(resp.text, resume_text_corpus, resume_bullets)
            if kw_hit:
                weighted_responsibility_keyword_hits += 1.0

            if self.embedding_matcher is not None and resume_bullets:
                score = self.embedding_matcher.best_match_score(resp.text, resume_bullets)
                weighted_responsibility_semantic_sum += score
            else:
                weighted_responsibility_semantic_sum += 0.0

        num_responsibilities = len(jd.responsibilities)

        skill_coverage_score = (
            weighted_skill_hits / weighted_requirement_total
            if weighted_requirement_total > 0
            else 0.0
        )

        keyword_alignment_score = (
            weighted_keyword_hits / weighted_keyword_total
            if weighted_keyword_total > 0
            else 0.0
        )

        responsibility_keyword_score = (
            weighted_responsibility_keyword_hits / num_responsibilities
            if num_responsibilities > 0
            else 0.0
        )

        ats_like_score = (
            0.45 * skill_coverage_score
            + 0.35 * keyword_alignment_score
            + 0.20 * responsibility_keyword_score
        )

        requirement_semantic_score = (
            weighted_requirement_semantic_sum / weighted_requirement_total
            if weighted_requirement_total > 0
            else 0.0
        )

        responsibility_semantic_score = (
            weighted_responsibility_semantic_sum / num_responsibilities
            if num_responsibilities > 0
            else 0.0
        )

        semantic_score = (
            0.70 * requirement_semantic_score
            + 0.30 * responsibility_semantic_score
        )

        blended_score = (
            0.55 * ats_like_score
            + 0.45 * semantic_score
        )

        semantic_gap = semantic_score - ats_like_score

        scores = MatchScores(
            skill_coverage_score=round(skill_coverage_score, 4),
            keyword_alignment_score=round(keyword_alignment_score, 4),
            responsibility_keyword_score=round(responsibility_keyword_score, 4),
            ats_like_score=round(ats_like_score, 4),
            requirement_semantic_score=round(requirement_semantic_score, 4),
            responsibility_semantic_score=round(responsibility_semantic_score, 4),
            semantic_score=round(semantic_score, 4),
            blended_score=round(blended_score, 4),
            semantic_gap=round(semantic_gap, 4),
        )

        report = AnalysisReportSchema(
            candidate_name=resume.candidate_name,
            job_title=jd.job_title,
            company_name=jd.company_name,
            matched_skills=unique_preserve_order(matched_skills),
            missing_skills=unique_preserve_order(missing_skills),
            matched_keywords=unique_preserve_order(matched_keywords_global),
            missing_keywords=unique_preserve_order(missing_keywords_global),
            scores=scores,
            requirement_matches=requirement_matches,
            priority_gaps=self._build_priority_gaps(requirement_matches),
            strongest_evidence_bullets=unique_preserve_order(strongest_evidence_bullets)[:5],
            overall_assessment=self._build_overall_assessment(scores),
        )

        return report

    def _evaluate_requirement(
        self,
        req: JDRequirement,
        resume: ResumeSchema,
        resume_skill_set: Set[str],
        resume_text_corpus: str,
        resume_bullets: List[ResumeBullet],
    ) -> RequirementEvaluation:
        if self._looks_like_degree_requirement(req.text):
            degree_ok = self._resume_has_degree_equivalent(resume, req.text)
            evidence_bullets: List[ResumeBullet] = []
            return RequirementEvaluation(
                matched_by_skill=degree_ok,
                matched_by_keyword=degree_ok,
                ats_matched=degree_ok,
                matched_by_semantics=degree_ok,
                semantic_score=1.0 if degree_ok else 0.0,
                matched_keywords=["degree_requirement"] if degree_ok else [],
                missing_keywords=[] if degree_ok else ["degree_requirement"],
                evidence_bullets=evidence_bullets,
            )

        keywords = tokenize_requirement(req)

        matched_keywords: List[str] = []
        missing_keywords: List[str] = []

        matched_by_skill = False
        matched_by_keyword = False

        for kw in keywords:
            in_skills = keyword_in_skill_set(kw, resume_skill_set)
            in_text = keyword_in_text(kw, resume_text_corpus)

            if in_skills or in_text:
                matched_keywords.append(kw)
            else:
                missing_keywords.append(kw)

            matched_by_skill = matched_by_skill or in_skills
            matched_by_keyword = matched_by_keyword or in_text

        ats_matched = matched_by_skill or matched_by_keyword

        keyword_evidence = self._find_keyword_evidence_bullets(keywords, resume_bullets)

        semantic_score = 0.0
        matched_by_semantics = False
        semantic_evidence: List[ResumeBullet] = []

        if self.embedding_matcher is not None and resume_bullets:
            top_matches = self.embedding_matcher.top_k_bullet_matches(
                query_text=req.text,
                bullets=resume_bullets,
                top_k=3,
            )

            if top_matches:
                semantic_score = top_matches[0].score
                matched_by_semantics = semantic_score >= self.semantic_threshold

                bullet_map = {b.id: b for b in resume_bullets}
                for match in top_matches:
                    if match.score >= self.semantic_evidence_threshold and match.bullet_id in bullet_map:
                        semantic_evidence.append(bullet_map[match.bullet_id])

        combined_evidence = self._merge_bullets(keyword_evidence, semantic_evidence, top_k=3)

        return RequirementEvaluation(
            matched_by_skill=matched_by_skill,
            matched_by_keyword=matched_by_keyword,
            ats_matched=ats_matched,
            matched_by_semantics=matched_by_semantics,
            semantic_score=semantic_score,
            matched_keywords=matched_keywords,
            missing_keywords=missing_keywords,
            evidence_bullets=combined_evidence,
        )

    def _find_keyword_evidence_bullets(
        self,
        keywords: List[str],
        resume_bullets: List[ResumeBullet],
        top_k: int = 3,
    ) -> List[ResumeBullet]:
        scored = []

        for bullet in resume_bullets:
            bullet_text = normalize_text(
                " ".join([bullet.text] + bullet.technologies + bullet.claims)
            )
            score = 0
            for kw in keywords:
                if keyword_in_text(kw, bullet_text):
                    score += 1
            if score > 0:
                scored.append((score, bullet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [bullet for _, bullet in scored[:top_k]]

    def _merge_bullets(
        self,
        bullets_a: List[ResumeBullet],
        bullets_b: List[ResumeBullet],
        top_k: int = 3,
    ) -> List[ResumeBullet]:
        seen = set()
        merged = []

        for bullet in bullets_a + bullets_b:
            if bullet.id not in seen:
                seen.add(bullet.id)
                merged.append(bullet)

        return merged[:top_k]

    def _responsibility_keyword_match(
        self,
        responsibility_text: str,
        resume_text_corpus: str,
        resume_bullets: List[ResumeBullet],
    ) -> bool:
        normalized_resp = normalize_token(responsibility_text)
        if not normalized_resp:
            return False

        if keyword_in_text(normalized_resp, resume_text_corpus):
            return True

        resp_tokens = {tok for tok in normalized_resp.split() if len(tok) > 3}
        for bullet in resume_bullets:
            bullet_text = normalize_text(
                " ".join([bullet.text] + bullet.technologies + bullet.claims)
            )
            bullet_tokens = set(bullet_text.split())
            if len(resp_tokens & bullet_tokens) >= 2:
                return True

        return False

    def _looks_like_degree_requirement(self, text: str) -> bool:
        t = text.lower()
        return any(x in t for x in ["bachelor", "master", "phd", "doctorate", "degree"])

    def _resume_has_degree_equivalent(self, resume: ResumeSchema, requirement_text: str) -> bool:
        req = requirement_text.lower()

        education_parts = []
        for edu in resume.education:
            for part in [edu.degree, edu.field_of_study, edu.institution]:
                if part:
                    education_parts.append(part.lower())

        joined = " ".join(education_parts)

        if "bachelor" in req:
            return any(x in joined for x in ["bachelor", "master", "phd", "doctor"])
        if "master" in req:
            return any(x in joined for x in ["master", "phd", "doctor"])
        if "phd" in req or "doctor" in req:
            return any(x in joined for x in ["phd", "doctor"])

        if "degree" in req:
            return any(x in joined for x in ["bachelor", "master", "phd", "doctor"])

        return False

    def _estimate_confidence(self, evaluation: RequirementEvaluation) -> float:
        score = 0.0
        if evaluation.matched_by_skill:
            score += 0.30
        if evaluation.matched_by_keyword:
            score += 0.25
        if evaluation.matched_by_semantics:
            score += 0.30
        score += min(len(evaluation.matched_keywords) * 0.05, 0.15)
        return round(min(score, 1.0), 3)

    def _build_rationale(self, req: JDRequirement, evaluation: RequirementEvaluation) -> str:
        if evaluation.ats_matched and evaluation.matched_by_semantics:
            return (
                f"Requirement '{req.text}' is supported by both ATS-style lexical evidence "
                f"and semantic similarity to resume bullets."
            )
        if evaluation.ats_matched:
            return (
                f"Requirement '{req.text}' is supported mainly by explicit lexical/keyword evidence."
            )
        if evaluation.matched_by_semantics:
            return (
                f"Requirement '{req.text}' is weak in explicit keywords but supported semantically by resume content."
            )
        return f"Requirement '{req.text}' has weak direct evidence in the resume."

    def _build_priority_gaps(
        self,
        requirement_matches: List[RequirementMatchResult],
    ) -> List[str]:
        gaps = []

        for item in requirement_matches:
            if item.importance == "required" and not item.ats_matched:
                gaps.append(item.requirement_text)

        if not gaps:
            for item in requirement_matches:
                if item.importance in {"required", "preferred"} and not (
                    item.ats_matched or item.matched_by_semantics
                ):
                    gaps.append(item.requirement_text)

        return unique_preserve_order(gaps)[:5]

    def _build_overall_assessment(self, scores: MatchScores) -> str:
        if scores.blended_score >= 0.75:
            level = "strong"
        elif scores.blended_score >= 0.50:
            level = "moderate"
        else:
            level = "limited"

        if scores.semantic_gap >= 0.15:
            gap_comment = (
                "Semantic alignment is noticeably stronger than ATS-style lexical alignment, "
                "suggesting the resume may undersell relevant experience in keyword terms."
            )
        elif scores.semantic_gap <= -0.15:
            gap_comment = (
                "Lexical alignment appears stronger than semantic alignment, suggesting the resume uses relevant wording "
                "but the underlying evidence may be weaker."
            )
        else:
            gap_comment = (
                "ATS-style and semantic alignment are broadly consistent."
            )

        return f"Overall alignment appears {level}. {gap_comment}"