🧠 Job Matcher (ATS + Semantic Hybrid)

A hybrid job matching system that combines ATS-style keyword matching with semantic understanding, designed to simulate real-world hiring pipelines while improving matching quality.

🚀 Motivation

Most Applicant Tracking Systems (ATS):
rely heavily on keyword matching
fail to capture semantic equivalence
often filter out strong candidates unfairly

This project is built to answer a practical question:
Can we model how ATS actually works — and improve it without breaking it?

🧠 Core Idea

We design a dual-channel matching system:
ATS channel → ensures real-world compatibility
Semantic channel → captures true capability

🔀 Final Score
Final Score = α * ATS_score + β * Semantic_score

This preserves deployability while improving accuracy.

🏗️ System Architecture
flowchart TD
    Resume --> Parser
    JD[Job Description] --> Parser

    Parser --> ATS[ATS Matcher<br/>(Keyword)]
    Parser --> Semantic[Semantic Matcher<br/>(Embeddings)]

    ATS --> Fusion[Score Fusion]
    Semantic --> Fusion

    Fusion --> Final[Final Score]
⚙️ Features

🔍 Hybrid ATS + semantic matching
🧩 Modular architecture (easy to extend)
📊 Explainable scoring
🧠 Embedding-based similarity (LLM-ready)
🛠️ Customizable skill taxonomy
📦 Installation
git clone https://github.com/rhonand/job_matcher.git
cd job-matcher
pip install -r requirements.txt

▶️ Usage
1. Parse Resume & Job Description
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

🧪 Design Philosophy
1. ATS is NOT wrong — just incomplete

We intentionally preserve keyword matching because:

real hiring pipelines depend on it
candidates must pass ATS before human review
2. Semantic ≠ Replacement

Pure semantic systems:

may overestimate relevance
may ignore hard constraints

👉 We combine, not replace

3. Explainability First

This system explicitly shows:

keyword matches (ATS view)
semantic similarity (model view)

This makes the system:

debuggable, inspectable, and closer to real hiring behavior

🔧 Customization
Adjust scoring weights
hybrid_match(resume, job, alpha=0.6, beta=0.4)
Extend skill taxonomy
KNOWN_SKILLS = [
    "C++",
    "PyTorch",
    "Sensor Fusion",
    "SLAM",
    "Distributed Systems"
]

📈 Future Work
🔄 RAG-based skill expansion
🧠 Context-aware matching (project-level reasoning)
📊 Multi-candidate ranking
🌐 Web UI for interactive analysis
🤖 LLM-based explanation generation
⚠️ Limitations
depends on skill extraction quality
semantic models may introduce false positives
not a replacement for human evaluation
💡 Why This Project Matters

This project models a real-world gap:

ATS systems filter based on keywords
Humans evaluate based on meaning

By bridging both:

candidates can optimize resumes more effectively
engineers can understand hiring systems deeply
systems can become more fair and accurate

🧷 Author Insight (Key Highlight)

A key observation behind this project:

Many hiring systems behave like deterministic filters,
while human evaluation is semantic and contextual

This project is an attempt to unify both views into a single system.