from typing import List, Optional, Literal
from pydantic import BaseModel, Field


class JDRequirement(BaseModel):
    id: str = Field(..., description="Unique requirement ID, e.g. req_0")
    text: str
    category: Literal[
        "programming_language",
        "framework",
        "ml_ai",
        "backend",
        "frontend",
        "cloud_devops",
        "data",
        "embedded_systems",
        "research",
        "soft_skill",
        "domain_knowledge",
        "tooling",
        "other"
    ]
    importance: Literal["required", "preferred", "nice_to_have"]
    keywords: List[str] = []


class JDResponsibility(BaseModel):
    id: str
    text: str
    keywords: List[str] = []


class JDSchema(BaseModel):
    job_title: Optional[str] = None
    company_name: Optional[str] = None
    seniority: Optional[str] = None
    employment_type: Optional[str] = None
    domain: Optional[str] = None
    summary: Optional[str] = None
    requirements: List[JDRequirement] = []
    responsibilities: List[JDResponsibility] = []
    keywords: List[str] = Field(default_factory=list, description="Global important JD keywords")