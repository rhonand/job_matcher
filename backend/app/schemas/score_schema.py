from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field


class MatchEvidence(BaseModel):
    requirement_id: str
    matched: bool
    confidence: float = 0.0
    evidence_bullet_ids: List[str] = Field(default_factory=list)
    evidence_texts: List[str] = Field(default_factory=list)
    rationale: Optional[str] = None


class RequirementMatchResult(BaseModel):
    requirement_id: str
    requirement_text: str
    importance: str

    matched_by_skill: bool = False
    matched_by_keyword: bool = False
    ats_matched: bool = False

    matched_by_semantics: bool = False
    semantic_score: float = 0.0

    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)

    evidence: Optional[MatchEvidence] = None


class MatchScores(BaseModel):
    skill_coverage_score: float = 0.0
    keyword_alignment_score: float = 0.0
    responsibility_keyword_score: float = 0.0

    ats_like_score: float = 0.0

    requirement_semantic_score: float = 0.0
    responsibility_semantic_score: float = 0.0
    semantic_score: float = 0.0

    blended_score: float = 0.0
    semantic_gap: float = 0.0


class RewriteSuggestion(BaseModel):
    bullet_id: str
    original_text: str
    rewritten_text: str
    reason: str
    supported_by_resume: bool = True
    unsupported_terms: List[str] = Field(default_factory=list)


class TailoredSummarySuggestion(BaseModel):
    original_summary: Optional[str] = None
    tailored_summary: str
    reason: str


class InterviewQuestionSuggestion(BaseModel):
    question: str
    rationale: str


class AnalysisMetadata(BaseModel):
    llm_model: Optional[str] = None
    embedding_model: Optional[str] = None
    prompt_version: Optional[str] = None
    processing_time_ms: Optional[int] = None


class AnalysisReportSchema(BaseModel):
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None

    matched_skills: List[str] = Field(default_factory=list)
    missing_skills: List[str] = Field(default_factory=list)

    matched_keywords: List[str] = Field(default_factory=list)
    missing_keywords: List[str] = Field(default_factory=list)

    scores: MatchScores = Field(default_factory=MatchScores)
    requirement_matches: List[RequirementMatchResult] = Field(default_factory=list)

    priority_gaps: List[str] = Field(default_factory=list)
    strongest_evidence_bullets: List[str] = Field(default_factory=list)

    rewrite_suggestions: List[RewriteSuggestion] = Field(default_factory=list)
    tailored_summary: Optional[TailoredSummarySuggestion] = None
    interview_questions: List[InterviewQuestionSuggestion] = Field(default_factory=list)

    overall_assessment: str = ""
    metadata: Optional[AnalysisMetadata] = None