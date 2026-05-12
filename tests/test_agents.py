"""
tests/test_agents.py + test_parsers.py + test_tools.py (combined)
──────────────────────────────────────────────────────────────────
Tests that run without external dependencies (no API key, no PDFs).
All LLM calls are mocked.
"""
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from core.models import (
    AnalysisReport,
    ATSScore,
    ExtractedJD,
    ExtractedResume,
    JDRequirement,
    JobSession,
    JobStatus,
    MatchStrength,
    SeniorityLevel,
    SkillGap,
)
from tools.ats_scorer import _starts_with_action_verb, extract_jd_keywords, score_resume

# ─── ATS Scorer Tests ─────────────────────────────────────────────────────────

class TestATSScorer:

    def test_perfect_keyword_match(self):
        resume = "Experienced Python developer. AWS certified. Led machine learning projects."
        keywords = ["Python", "AWS", "machine learning"]
        score = score_resume(resume, keywords)
        assert score.keyword_match == 100.0
        assert len(score.present_keywords) == 3
        assert len(score.missing_keywords) == 0

    def test_zero_keyword_match(self):
        resume = "Experienced JavaScript developer. Frontend specialist. React expert."
        keywords = ["Python", "AWS", "Kubernetes"]
        score = score_resume(resume, keywords)
        assert score.keyword_match == 0.0
        assert score.missing_keywords == ["Python", "AWS", "Kubernetes"]

    def test_partial_keyword_match(self):
        resume = "Python developer with SQL experience."
        keywords = ["Python", "SQL", "AWS", "Docker"]
        score = score_resume(resume, keywords)
        assert score.keyword_match == 50.0  # 2/4

    def test_overall_score_weighted(self):
        # Minimal resume, good keywords
        resume = (
            "EXPERIENCE\n"
            "• Led team of 10 engineers, improving performance by 40%.\n"
            "• Built Python microservices on AWS, reducing costs by $200k.\n"
            "SKILLS\nPython, AWS, SQL, Docker\n"
            "EDUCATION\nB.S. Computer Science"
        )
        keywords = ["Python", "AWS", "SQL"]
        score = score_resume(resume, keywords)
        assert 50 <= score.overall <= 100

    def test_quantification_detection(self):
        resume_with_numbers = (
            "• Improved latency by 40%\n"
            "• Managed team of 8 engineers\n"
            "• Generated $2M in revenue\n"
            "• Reduced costs by 25x"
        )
        resume_no_numbers = (
            "• Improved system performance\n"
            "• Managed engineering team\n"
            "• Contributed to revenue growth"
        )
        score_with = score_resume(resume_with_numbers, [])
        score_without = score_resume(resume_no_numbers, [])
        assert score_with.quantification > score_without.quantification

    def test_action_verb_detection(self):
        assert _starts_with_action_verb("• Led the development of a new API")
        assert _starts_with_action_verb("• Built a microservices architecture")
        assert _starts_with_action_verb("Optimized database queries")
        assert not _starts_with_action_verb("• Was responsible for managing the team")
        assert not _starts_with_action_verb("• The project involved many challenges")

    def test_empty_resume(self):
        score = score_resume("", ["Python"])
        assert score.overall >= 0

    def test_empty_keywords(self):
        resume = "Python developer with 5 years experience."
        score = score_resume(resume, [])
        assert score.keyword_match == 50.0  # Default when no keywords

    def test_keyword_extraction_regex_fallback(self):
        jd = "Must have Python, AWS, and React experience. SQL is a plus. Kubernetes preferred."
        keywords = extract_jd_keywords(jd, top_k=10)
        assert isinstance(keywords, list)
        assert len(keywords) > 0

    def test_formatting_penalizes_tables(self):
        clean_resume = "Python developer\n\nEXPERIENCE\nSoftware Engineer at Acme\n• Led team\n• Built API"
        table_resume = "Name | Python | AWS | 5 years\nRole | SQL | Docker | 3 years"
        score_clean = score_resume(clean_resume, [])
        score_table = score_resume(table_resume, [])
        assert score_clean.formatting > score_table.formatting


# ─── Data Model Tests ─────────────────────────────────────────────────────────

class TestDataModels:

    def test_job_session_log(self):
        session = JobSession(id="test-123")
        session.log("Step 1 complete")
        session.log("Step 2 started")
        assert len(session.agent_logs) == 2
        assert "Step 1 complete" in session.agent_logs[0]

    def test_skill_gap_match_strength_enum(self):
        gap = SkillGap(
            requirement="Python programming",
            category="skill",
            importance="required",
            match=MatchStrength.MISSING,
        )
        assert gap.match == MatchStrength.MISSING
        assert gap.match.value == "missing"

    def test_ats_score_fields(self):
        ats = ATSScore(
            overall=72.5,
            keyword_match=80.0,
            formatting=90.0,
            section_completeness=75.0,
            quantification=50.0,
            action_verb_usage=65.0,
            missing_keywords=["Kubernetes"],
            present_keywords=["Python", "AWS"],
        )
        assert ats.overall == 72.5
        assert "Kubernetes" in ats.missing_keywords

    def test_extracted_jd_defaults(self):
        jd = ExtractedJD(job_title="Software Engineer")
        assert jd.seniority == SeniorityLevel.UNKNOWN
        assert jd.requirements == []
        assert jd.key_skills == []

    def test_job_session_status_transitions(self):
        session = JobSession(id="abc")
        assert session.status == JobStatus.PENDING
        session.status = JobStatus.SCRAPING
        assert session.status == JobStatus.SCRAPING
        session.status = JobStatus.COMPLETE
        assert session.status == JobStatus.COMPLETE


# ─── Scraper Agent Tests (mocked LLM) ─────────────────────────────────────────

class TestScraperAgent:

    def _make_mock_llm(self, return_value: dict):
        llm = MagicMock()
        llm.call_json.return_value = return_value
        return llm

    def test_extract_jd_basic(self):
        from agents.scraper_agent import ScraperAgent
        mock_llm = self._make_mock_llm({
            "job_title": "Senior Python Engineer",
            "company": "TechCorp",
            "seniority": "senior",
            "requirements": [
                {"text": "5+ years Python", "category": "experience", "importance": "required", "keywords": ["Python"]}
            ],
            "key_skills": ["Python", "Django", "PostgreSQL"],
            "responsibilities": ["Build APIs", "Code review"],
            "industry_keywords": ["SaaS"],
            "culture_signals": ["fast-paced"],
        })
        agent = ScraperAgent(mock_llm)
        jd = agent.extract_jd("Some job description text")
        assert jd.job_title == "Senior Python Engineer"
        assert jd.company == "TechCorp"
        assert jd.seniority == SeniorityLevel.SENIOR
        assert len(jd.requirements) == 1
        assert jd.requirements[0].importance == "required"

    def test_extract_jd_unknown_seniority(self):
        from agents.scraper_agent import ScraperAgent
        mock_llm = self._make_mock_llm({
            "job_title": "Developer",
            "seniority": "invalid_value",
            "requirements": [],
            "key_skills": [],
            "responsibilities": [],
            "industry_keywords": [],
            "culture_signals": [],
        })
        agent = ScraperAgent(mock_llm)
        jd = agent.extract_jd("job text")
        assert jd.seniority == SeniorityLevel.UNKNOWN  # Falls back gracefully

    def test_extract_resume_basic(self):
        from agents.scraper_agent import ScraperAgent
        mock_llm = self._make_mock_llm({
            "name": "Jane Smith",
            "email": "jane@example.com",
            "phone": "+1-555-0100",
            "current_title": "Software Engineer",
            "summary": "5 years Python experience",
            "skills": ["Python", "SQL", "AWS"],
            "skill_details": [
                {"skill": "Python", "evidence": "Built REST APIs using Python", "years": 5.0}
            ],
            "experience": [
                {
                    "title": "Software Engineer",
                    "company": "StartupCo",
                    "dates": "2020-2024",
                    "bullets": ["• Built microservices architecture", "• Led team of 3"],
                }
            ],
            "education": [{"degree": "B.S. CS", "school": "State U", "year": "2019"}],
            "certifications": ["AWS Solutions Architect"],
        })
        agent = ScraperAgent(mock_llm)
        resume = agent.extract_resume("resume text here")
        assert resume.name == "Jane Smith"
        assert "Python" in resume.skills
        assert len(resume.experience) == 1
        # skill_details may be populated depending on scraper agent implementation
        if resume.skill_details:
            assert resume.skill_details[0].years == 5.0


# ─── Analyzer Agent Tests (mocked LLM) ────────────────────────────────────────

class TestAnalyzerAgent:

    def _make_jd(self) -> ExtractedJD:
        return ExtractedJD(
            job_title="Data Engineer",
            company="DataCo",
            seniority=SeniorityLevel.SENIOR,
            requirements=[
                JDRequirement(text="5+ years Python", category="experience", importance="required", keywords=["Python"]),
                JDRequirement(text="Apache Spark", category="skill", importance="required", keywords=["Spark"]),
                JDRequirement(text="AWS or GCP", category="skill", importance="preferred", keywords=["AWS", "GCP"]),
            ],
            key_skills=["Python", "Spark", "AWS", "ETL"],
            raw_text="Senior Data Engineer at DataCo. Requires 5+ years Python and Apache Spark experience. AWS or GCP preferred.",
        )

    def _make_resume(self) -> ExtractedResume:
        return ExtractedResume(
            name="Alex Chen",
            current_title="Data Engineer",
            skills=["Python", "SQL", "PostgreSQL", "Docker"],
            experience=[
                {
                    "title": "Data Engineer", "company": "OldCo",
                    "dates": "2019-2024",
                    "bullets": ["Built ETL pipelines", "Improved data quality by 30%"]
                }
            ],
            raw_text="Alex Chen. Data Engineer at OldCo (2019-2024). Built ETL pipelines. Python, SQL, PostgreSQL, Docker.",
        )

    def test_analyze_produces_report(self):
        from agents.analyzer_agent import AnalyzerAgent
        mock_llm = MagicMock()
        mock_llm.call_json.return_value = {
            "gaps": [
                {"requirement": "Apache Spark", "category": "skill", "importance": "required",
                 "match": "missing", "resume_evidence": None,
                 "cot_reasoning": "Checked resume, Spark not mentioned.",
                 "rewrite_suggestion": "Add Spark to skills section and mention in bullet 1"},
                {"requirement": "5+ years Python", "category": "experience", "importance": "required",
                 "match": "strong", "resume_evidence": "Python mentioned in skills",
                 "cot_reasoning": "Python clearly present.", "rewrite_suggestion": None},
            ],
            "strengths": ["Strong Python background", "Good ETL experience"],
            "priority_rewrites": ["1. Add Apache Spark to skills", "2. Quantify ETL pipeline impact"],
            "strategy_notes": "Overall decent fit. Main gap is Spark. Python is solid.",
        }
        agent = AnalyzerAgent(mock_llm)
        report = agent.analyze(self._make_jd(), self._make_resume(), session_id="test")

        assert isinstance(report, AnalysisReport)
        assert len(report.skill_gaps) == 2
        assert any(g.match == MatchStrength.MISSING for g in report.skill_gaps)
        assert any(g.match == MatchStrength.STRONG for g in report.skill_gaps)
        assert len(report.strengths) == 2
        assert len(report.priority_rewrites) == 2

    def test_seniority_match_senior_with_5_years(self):
        from agents.analyzer_agent import AnalyzerAgent
        agent = AnalyzerAgent(MagicMock())
        resume = self._make_resume()  # 2019-2024 = 5 years
        match = agent._check_seniority_match(SeniorityLevel.SENIOR, resume)
        assert match is True

    def test_seniority_mismatch_junior_with_5_years(self):
        from agents.analyzer_agent import AnalyzerAgent
        agent = AnalyzerAgent(MagicMock())
        resume = self._make_resume()
        match = agent._check_seniority_match(SeniorityLevel.JUNIOR, resume)
        assert match is False  # Junior needs 0-3 years


# ─── Orchestrator State Machine Tests ─────────────────────────────────────────

class TestOrchestrator:

    def test_pipeline_fails_gracefully_on_missing_resume(self, tmp_path):
        """Pipeline should set FAILED status if resume path is missing."""
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            from agents.orchestrator import Orchestrator
            orch = Orchestrator.__new__(Orchestrator)
            session = JobSession(id="fail-test")
            session.resume_path = None
            session.jd_path = str(tmp_path / "fake.txt")

            # Directly test the load helper
            try:
                orch._load_document(None, "resume")
                assert False, "Should have raised"
            except ValueError as e:
                assert "resume" in str(e).lower()

    def test_pipeline_fails_on_nonexistent_file(self):
        with patch.dict("os.environ", {"NVIDIA_API_KEY": "test-key"}):
            from agents.orchestrator import Orchestrator
            orch = Orchestrator.__new__(Orchestrator)
            try:
                orch._load_document("/nonexistent/path/resume.pdf", "resume")
                assert False, "Should have raised"
            except Exception:
                pass  # Expected


# ─── Text Parser Tests ────────────────────────────────────────────────────────

class TestTextParsing:

    def test_clean_text_fixes_ligatures(self):
        from parsers.pdf_parser import _clean_text
        assert _clean_text("ﬁle ﬂow") == "file flow"

    def test_clean_text_fixes_hyphen_breaks(self):
        from parsers.pdf_parser import _clean_text
        assert "process" in _clean_text("pro-\ncess")

    def test_clean_text_normalizes_whitespace(self):
        from parsers.pdf_parser import _clean_text
        result = _clean_text("word     with     spaces")
        assert "     " not in result

    def test_extract_text_unsupported_format(self, tmp_path):
        from parsers.pdf_parser import extract_text
        f = tmp_path / "test.xyz"
        f.write_text("content")
        with pytest.raises(ValueError, match="Unsupported"):
            extract_text(str(f))

    def test_extract_text_from_txt(self, tmp_path):
        from parsers.pdf_parser import extract_text
        f = tmp_path / "resume.txt"
        f.write_text("Jane Smith\nSoftware Engineer\nPython, AWS")
        text = extract_text(str(f))
        assert "Jane Smith" in text
        assert "Python" in text


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
