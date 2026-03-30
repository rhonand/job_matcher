Job Matcher (ATS + Semantic Hybrid)

A hybrid job matching system that combines rule-based ATS-style keyword matching with semantic similarity modeling, designed to simulate real-world hiring pipelines while improving matching quality beyond traditional ATS systems.

Motivation

Most Applicant Tracking Systems (ATS) rely heavily on keyword matching, which leads to:

Missing strong candidates due to wording differences
Overweighting superficial keyword overlap
Ignoring semantic equivalence (e.g., "distributed system" vs "microservices architecture")

This project aims to:

Reproduce ATS behavior (for realism)
Enhance it with semantic understanding
Bridge the gap between machine filtering and human judgment

Core Idea

We use a dual-channel matching system:

1. ATS Channel (Deterministic)
Exact keyword matching
Skill dictionary / known terms
Weighted scoring based on frequency and importance
2. Semantic Channel (Intelligent)
Embedding-based similarity
Captures meaning beyond exact wording
Handles paraphrases and domain variation

Final Score

A weighted combination:

Final Score = α * ATS_score + β * Semantic_score

Where:

ATS ensures real-world compatibility
Semantic ensures true capability recognition
System Architecture
                ┌──────────────┐
                │   Resume     │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │   Parser     │
                └──────┬───────┘
                       │
        ┌──────────────┴──────────────┐
        │                             │
┌───────▼────────┐         ┌──────────▼────────┐
│ ATS Matcher    │         │ Semantic Matcher  │
│ (Keyword)      │         │ (Embeddings)      │
└───────┬────────┘         └──────────┬────────┘
        │                             │
        └──────────────┬──────────────┘
                       │
                ┌──────▼───────┐
                │ Score Fusion │
                └──────┬───────┘
                       │
                ┌──────▼───────┐
                │ Final Score  │
                └──────────────┘
Features
Hybrid ATS + semantic matching
Modular architecture (easy to extend)
Explainable scoring (keyword vs semantic contribution)
Embedding-based similarity (LLM-ready)
Customizable skill taxonomy
Installation
git clone https://github.com/rhonand/job-matcher.git
cd job-matcher

pip install -r requirements.txt
Usage
1. Parse Resume & JD
from matcher import parse_resume, parse_job

resume = parse_resume("resume.pdf")
job = parse_job("job_description.txt")

2. Run Matching
from matcher import hybrid_match

result = hybrid_match(resume, job)

print(result)

3. Example Output
{
  "final_score": 0.78,
  "ats_score": 0.65,
  "semantic_score": 0.89,
  "matched_keywords": [
    "Python",
    "Distributed Systems",
    "Computer Vision"
  ],
  "missing_keywords": [
    "Kubernetes",
    "AWS"
  ]
}

Design Philosophy
1. ATS is NOT wrong — just incomplete

We intentionally preserve keyword matching because:

Real hiring pipelines depend on it
Candidates must pass ATS before humans see them

2. Semantic ≠ Replacement, but Enhancement

Pure semantic matching can:

Overestimate relevance
Miss required hard constraints

So we combine, not replace.

3. Explainability Matters

Unlike black-box scoring systems, this project:

Shows why a score is high/low
Separates missing keywords vs semantic gaps
🔧 Customization
Adjust scoring weights
hybrid_match(resume, job, alpha=0.6, beta=0.4)
Add domain-specific skills
KNOWN_SKILLS = [
    "C++",
    "PyTorch",
    "Sensor Fusion",
    "SLAM",
    "Distributed Systems"
]

Future Work
RAG-based skill expansion
Context-aware matching (project-level reasoning)
Ranking multiple candidates
Web UI for interactive analysis
LLM-based explanation generation
Limitations
Depends on quality of skill extraction
Semantic model may produce false positives
Not a replacement for human evaluation

Why This Project Matters

This is not just a toy project — it reflects a real problem:

The gap between what ATS filters and what engineers can actually do

By modeling both sides, this system:

Helps candidates optimize resumes
Helps engineers understand hiring systems
Moves toward fairer evaluation