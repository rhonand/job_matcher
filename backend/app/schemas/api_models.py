from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field

from backend.app.schemas.jd_schema import JDSchema
from backend.app.schemas.score_schema import AnalysisReportSchema
from backend.app.schemas.resume_schema import ResumeSchema


class ParseResumeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)


class ParseJDRequest(BaseModel):
    jd_text: str = Field(..., min_length=1)


class MatchRequest(BaseModel):
    resume: ResumeSchema
    jd: JDSchema


class SuggestRequest(BaseModel):
    resume: ResumeSchema
    jd: JDSchema
    match_report: AnalysisReportSchema
    max_rewrite_suggestions: int = 3


class AnalyzeRequest(BaseModel):
    resume_text: str = Field(..., min_length=1)
    jd_text: str = Field(..., min_length=1)
    top_k_bullets: int = 5
    max_rewrite_suggestions: int = 3


class ExtractedTextResponse(BaseModel):
    text: str


class ErrorResponse(BaseModel):
    detail: str


class ParseResumeResponse(BaseModel):
    parsed_resume: ResumeSchema


class ParseJDResponse(BaseModel):
    parsed_jd: JDSchema


class AnalyzeResponse(BaseModel):
    report: AnalysisReportSchema