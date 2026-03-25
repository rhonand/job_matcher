from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class ResumeContact(BaseModel):
    email: Optional[str] = None
    phone: Optional[str] = None
    location: Optional[str] = None
    linkedin: Optional[str] = None
    github: Optional[str] = None
    website: Optional[str] = None


class ResumeBullet(BaseModel):
    id: str = Field(..., description="Unique bullet ID, e.g. exp_0_bullet_1")
    text: str
    technologies: List[str] = []
    claims: List[str] = Field(default_factory=list, description="Short factual claims extracted from the bullet")
    section: Literal["experience", "project", "research", "other"]


class ResumeExperienceItem(BaseModel):
    id: str
    title: Optional[str] = None
    organization: Optional[str] = None
    location: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bullets: List[ResumeBullet] = []


class ResumeProjectItem(BaseModel):
    id: str
    name: str
    role: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None
    bullets: List[ResumeBullet] = []


class ResumeEducationItem(BaseModel):
    degree: Optional[str] = None
    field_of_study: Optional[str] = None
    institution: Optional[str] = None
    start_date: Optional[str] = None
    end_date: Optional[str] = None


class ResumeSchema(BaseModel):
    candidate_name: Optional[str] = None
    contact: Optional[ResumeContact] = None
    summary: Optional[str] = None
    raw_skills: List[str] = Field(default_factory=list, description="Skills explicitly listed in resume")
    inferred_skills: List[str] = Field(default_factory=list, description="Skills inferred from bullets and content")
    experience: List[ResumeExperienceItem] = []
    projects: List[ResumeProjectItem] = []
    education: List[ResumeEducationItem] = []
    certifications: List[str] = []
    languages: List[str] = []