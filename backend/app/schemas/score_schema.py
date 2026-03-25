from typing import List, Optional
from pydantic import BaseModel


class MatchEvidence(BaseModel):
    requirement_id: str
    matched: bool
    confidence: float = 0.0
    evidence_bullet_ids: List[str] = []
    evidence_texts: List[str] = []
    rationale: Optional[str] = None

class RequirementMatchResult(BaseModel):
    requirement_id: str
    requirement_text: str
    importance: str
    matched_by_skill: bool = False
    matched_by_semantics: bool = False
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []
    evidence: Optional[MatchEvidence] = None

class MatchScores(BaseModel):
    skill_coverage_score: float = 0.0
    keyword_alignment_score: float = 0.0
    semantic_similarity_score: float = 0.0
    responsibility_coverage_score: float = 0.0
    overall_score: float = 0.0

class RewriteSuggestion(BaseModel):
    bullet_id: str
    original_text: str
    rewritten_text: str
    reason: str
    supported_by_resume: bool = True
    unsupported_terms: List[str] = []


class TailoredSummarySuggestion(BaseModel):
    original_summary: Optional[str] = None
    tailored_summary: str
    reason: str


class InterviewQuestionSuggestion(BaseModel):
    question: str
    rationale: str

class AnalysisReportSchema(BaseModel):
    candidate_name: Optional[str] = None
    job_title: Optional[str] = None
    company_name: Optional[str] = None

    matched_skills: List[str] = []
    missing_skills: List[str] = []
    matched_keywords: List[str] = []
    missing_keywords: List[str] = []

    scores: MatchScores
    requirement_matches: List[RequirementMatchResult] = []

    priority_gaps: List[str] = []
    strongest_evidence_bullets: List[str] = []

    rewrite_suggestions: List[RewriteSuggestion] = []
    tailored_summary: Optional[TailoredSummarySuggestion] = None
    interview_questions: List[InterviewQuestionSuggestion] = []

    overall_assessment: str