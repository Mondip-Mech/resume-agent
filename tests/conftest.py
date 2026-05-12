"""
tests/conftest.py
──────────────────
Shared pytest fixtures used across all test modules.
Every agent test imports from here so boilerplate stays in one place.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    AnalysisReport,
    ATSScore,
    CritiqueReport,
    ExtractedJD,
    ExtractedResume,
    JDRequirement,
    MatchStrength,
    RewrittenResume,
    SeniorityLevel,
    SkillGap,
)

# ─── LLM mock ────────────────────────────────────────────────────────────────

@pytest.fixture
def mock_llm():
    """
    A MagicMock that looks like LLMClient.
    Override call_json / call per test:
        mock_llm.call_json.return_value = {...}
        mock_llm.call.return_value = "some text"
    """
    llm = MagicMock()
    llm.call_json.return_value = {}
    llm.call.return_value = "Mock LLM response"
    llm.token_usage = {"input": 0, "output": 0, "total": 0}
    return llm


# ─── Raw text fixtures ────────────────────────────────────────────────────────

@pytest.fixture
def sample_jd_text() -> str:
    return """
    Data Scientist — Acme Analytics

    We are looking for a Senior Data Scientist to join our team.

    Requirements:
    - 5+ years of Python experience (required)
    - Proficiency in SQL and BigQuery (required)
    - Machine Learning: TensorFlow or PyTorch (required)
    - Experience with AWS (S3, EC2, SageMaker) (required)
    - A/B Testing and statistical analysis (preferred)
    - Docker and Kubernetes (preferred)
    - Tableau or Looker for dashboards (nice to have)

    Responsibilities:
    - Build and deploy ML models to production
    - Write scalable ETL pipelines using Spark or Airflow
    - Collaborate with engineering teams on MLOps

    Culture: fast-paced, data-driven, collaborative, ownership mindset.
    """


@pytest.fixture
def sample_resume_text() -> str:
    return """
    Jane Smith
    jane.smith@email.com | +1-555-0100 | LinkedIn: janesmith

    SUMMARY
    Data Scientist with 6 years of experience building machine learning models
    and data pipelines. Expert in Python and SQL. AWS certified.

    SKILLS
    Python, SQL, PostgreSQL, Pandas, NumPy, scikit-learn, TensorFlow,
    Docker, Git, Tableau, Jupyter

    EXPERIENCE
    Senior Data Scientist | TechCorp | 2021–2024
    - Built XGBoost churn prediction model achieving 92% accuracy, reducing churn by 18%
    - Designed ETL pipelines processing 5TB daily data with 99.9% uptime
    - Led A/B tests for 3 product features, improving conversion rate by 12%
    - Mentored team of 4 junior data scientists

    Data Analyst | DataInc | 2018–2021
    - Developed SQL dashboards in Tableau serving 200+ internal stakeholders
    - Automated 15 manual reporting workflows, saving 40 hours/week
    - Analyzed customer behavior datasets of 10M+ rows using Python and Pandas

    EDUCATION
    M.S. Data Science | State University | 2018
    B.S. Computer Science | State University | 2016

    CERTIFICATIONS
    AWS Certified Machine Learning Specialty
    """


# ─── Pydantic model fixtures ──────────────────────────────────────────────────

@pytest.fixture
def sample_extracted_jd() -> ExtractedJD:
    return ExtractedJD(
        job_title="Data Scientist",
        company="Acme Analytics",
        seniority=SeniorityLevel.SENIOR,
        requirements=[
            JDRequirement(text="5+ years Python", category="experience",
                          importance="required", keywords=["Python"]),
            JDRequirement(text="SQL and BigQuery", category="skill",
                          importance="required", keywords=["SQL", "BigQuery"]),
            JDRequirement(text="TensorFlow or PyTorch", category="skill",
                          importance="required", keywords=["TensorFlow", "PyTorch"]),
            JDRequirement(text="AWS experience", category="skill",
                          importance="required", keywords=["AWS"]),
            JDRequirement(text="Docker and Kubernetes", category="skill",
                          importance="preferred", keywords=["Docker", "Kubernetes"]),
        ],
        key_skills=["Python", "SQL", "BigQuery", "TensorFlow", "AWS", "Docker"],
        responsibilities=["Build ML models", "Write ETL pipelines", "MLOps"],
        culture_signals=["fast-paced", "data-driven", "collaborative"],
        raw_text="Data Scientist at Acme Analytics. Python, SQL, TensorFlow, AWS required.",
    )


@pytest.fixture
def sample_extracted_resume() -> ExtractedResume:
    return ExtractedResume(
        name="Jane Smith",
        email="jane.smith@email.com",
        phone="+1-555-0100",
        current_title="Senior Data Scientist",
        summary="Data Scientist with 6 years experience in Python and ML.",
        skills=["Python", "SQL", "TensorFlow", "Docker", "Tableau"],
        experience=[
            {
                "title": "Senior Data Scientist",
                "company": "TechCorp",
                "dates": "2021-2024",
                "bullets": [
                    "Built XGBoost churn prediction model achieving 92% accuracy",
                    "Designed ETL pipelines processing 5TB daily data",
                    "Led A/B tests improving conversion rate by 12%",
                ],
            },
            {
                "title": "Data Analyst",
                "company": "DataInc",
                "dates": "2018-2021",
                "bullets": [
                    "Developed SQL dashboards in Tableau",
                    "Automated 15 manual reporting workflows saving 40 hours/week",
                ],
            },
        ],
        education=[
            {"degree": "M.S. Data Science", "school": "State University", "year": "2018"}
        ],
        certifications=["AWS Certified Machine Learning Specialty"],
        raw_text="Jane Smith. Senior Data Scientist at TechCorp. Python, SQL, TensorFlow, Docker.",
    )


@pytest.fixture
def sample_ats_score() -> ATSScore:
    return ATSScore(
        overall=68.5,
        keyword_match=70.0,
        formatting=90.0,
        section_completeness=100.0,
        quantification=60.0,
        action_verb_usage=75.0,
        missing_keywords=["BigQuery", "Kubernetes", "PyTorch"],
        present_keywords=["Python", "SQL", "TensorFlow", "Docker"],
    )


@pytest.fixture
def sample_analysis(sample_extracted_jd, sample_ats_score) -> AnalysisReport:
    return AnalysisReport(
        job_id="test-session-001",
        match_score=72.0,
        ats_score=sample_ats_score,
        seniority_match=True,
        skill_gaps=[
            SkillGap(requirement="BigQuery", category="skill", importance="required",
                     match=MatchStrength.MISSING,
                     rewrite_suggestion="Add BigQuery to skills section"),
            SkillGap(requirement="Python", category="skill", importance="required",
                     match=MatchStrength.STRONG, resume_evidence="Python in skills"),
            SkillGap(requirement="Docker", category="skill", importance="preferred",
                     match=MatchStrength.PARTIAL, rewrite_suggestion="Expand Docker experience"),
        ],
        strengths=["Strong Python background", "Quantified achievements"],
        priority_rewrites=["Add BigQuery to skills", "Mention Kubernetes"],
        strategy_notes="Strong candidate. Main gap is BigQuery.",
    )


@pytest.fixture
def sample_rewritten_resume(sample_ats_score) -> RewrittenResume:
    improved_ats = ATSScore(
        overall=81.0,
        keyword_match=90.0,
        formatting=90.0,
        section_completeness=100.0,
        quantification=75.0,
        action_verb_usage=85.0,
        missing_keywords=["Kubernetes"],
        present_keywords=["Python", "SQL", "BigQuery", "TensorFlow", "Docker"],
    )
    return RewrittenResume(
        job_id="test-session-001",
        full_text=(
            "Jane Smith\njane.smith@email.com\n\n"
            "SUMMARY\nSenior Data Scientist with 6 years experience. "
            "Expert in Python, SQL, BigQuery, TensorFlow, and AWS.\n\n"
            "SKILLS\nPython, SQL, BigQuery, TensorFlow, AWS, Docker, Tableau\n\n"
            "EXPERIENCE\nSenior Data Scientist | TechCorp | 2021–2024\n"
            "- Built XGBoost model achieving 92% accuracy, reducing churn by 18%\n"
            "- Designed BigQuery ETL pipelines processing 5TB daily data\n"
        ),
        added_keywords=["BigQuery", "MLOps"],
        removed_content=[],
        new_ats_score=improved_ats,
    )


@pytest.fixture
def sample_critique() -> CritiqueReport:
    return CritiqueReport(
        passed=True,
        overall_score=78.0,
        keyword_coverage=85.0,
        star_compliance=75.0,
        issues=[],
        praise=["Strong summary", "BigQuery added correctly"],
        fix_instructions="",
    )


# ─── Memory fixture ───────────────────────────────────────────────────────────

@pytest.fixture
def tmp_memory(tmp_path):
    """AgentMemory backed by a temp directory — isolated per test."""
    from core.memory import AgentMemory
    return AgentMemory(persist_dir=str(tmp_path / "test_memory"))
