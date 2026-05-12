"""
tests/test_orchestrator.py
───────────────────────────
Integration tests for the Orchestrator pipeline.

All LLM calls and agent methods are mocked — no NVIDIA API key needed.
Tests verify the state machine, error isolation, and fallback strategies.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import (
    ApplicationPackage,
    CoverLetter,
    JobSession,
    JobStatus,
)

# ─── Helpers ─────────────────────────────────────────────────────────────────

def _make_orch(tmp_path):
    """Build an Orchestrator with all agents mocked, skipping __init__."""
    with patch.dict("os.environ", {
        "NVIDIA_API_KEY": "test-key",
        "OUTPUT_DIR": str(tmp_path),
    }):
        from agents.orchestrator import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)

        # Wire up a mock LLM and memory
        orch.llm = MagicMock()
        orch.llm.token_usage = {"input": 0, "output": 0, "total": 0}
        orch.memory = MagicMock()
        orch.memory.get_style_context.return_value = ""
        orch.memory.extract_and_store_bullets.return_value = []

        # Mock all agents
        orch.scraper   = MagicMock()
        orch.analyzer  = MagicMock()
        orch.critiquer = MagicMock()
        orch.rewriter  = MagicMock()
        orch.app_agent = MagicMock()
        orch.app_agent.output_dir = str(tmp_path)

        return orch


def _make_session(tmp_path, resume_text="Jane Smith\nPython developer", jd_text="Data Scientist role") -> JobSession:
    """Write minimal files and return a configured JobSession."""
    resume_file = tmp_path / "resume.txt"
    resume_file.write_text(resume_text, encoding="utf-8")
    jd_file = tmp_path / "jd.txt"
    jd_file.write_text(jd_text, encoding="utf-8")

    session = JobSession(id="test-session-001")
    session.resume_path = str(resume_file)
    session.jd_path = str(jd_file)
    return session


# ─── Document loading ─────────────────────────────────────────────────────────

class TestLoadDocument:

    def test_raises_on_none_path(self, tmp_path):
        from agents.orchestrator import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        with pytest.raises(ValueError, match="resume"):
            orch._load_document(None, "resume")

    def test_raises_on_nonexistent_file(self, tmp_path):
        from agents.orchestrator import Orchestrator
        orch = Orchestrator.__new__(Orchestrator)
        with pytest.raises(Exception):
            orch._load_document(str(tmp_path / "missing.pdf"), "resume")

    def test_loads_txt_file(self, tmp_path):
        from agents.orchestrator import Orchestrator
        f = tmp_path / "test.txt"
        f.write_text("Hello World", encoding="utf-8")
        orch = Orchestrator.__new__(Orchestrator)
        text = orch._load_document(str(f), "jd")
        assert "Hello World" in text


# ─── Validation helpers ───────────────────────────────────────────────────────

class TestValidations:

    def test_validate_scraper_output_warns_on_empty_requirements(self, tmp_path, sample_extracted_jd, sample_extracted_resume):
        orch = _make_orch(tmp_path)
        session = JobSession(id="v-test")
        sample_extracted_jd.requirements = []
        session.extracted_jd = sample_extracted_jd
        session.extracted_resume = sample_extracted_resume
        orch._validate_scraper_output(session)
        assert any("requirements" in log.lower() or "WARNING" in log for log in session.agent_logs)

    def test_validate_scraper_output_warns_on_empty_skills(self, tmp_path, sample_extracted_jd, sample_extracted_resume):
        orch = _make_orch(tmp_path)
        session = JobSession(id="v-test2")
        sample_extracted_resume.skills = []
        sample_extracted_resume.experience = []
        session.extracted_jd = sample_extracted_jd
        session.extracted_resume = sample_extracted_resume
        orch._validate_scraper_output(session)
        assert any("WARNING" in log for log in session.agent_logs)

    def test_validate_rewrite_warns_if_name_missing(self, tmp_path, sample_extracted_resume, sample_rewritten_resume):
        orch = _make_orch(tmp_path)
        session = JobSession(id="v-test3")
        session.extracted_resume = sample_extracted_resume
        # Name is "Jane Smith" but rewritten text doesn't contain it
        sample_rewritten_resume.full_text = "Some generic resume without the candidate name"
        session.rewritten_resume = sample_rewritten_resume
        orch._validate_rewrite(session)
        assert any("WARNING" in log for log in session.agent_logs)

    def test_validate_rewrite_warns_on_short_resume(self, tmp_path, sample_extracted_resume, sample_rewritten_resume):
        orch = _make_orch(tmp_path)
        session = JobSession(id="v-test4")
        session.extracted_resume = sample_extracted_resume
        sample_rewritten_resume.full_text = "Too short"  # < 300 chars
        session.rewritten_resume = sample_rewritten_resume
        orch._validate_rewrite(session)
        assert any("short" in log.lower() or "WARNING" in log for log in session.agent_logs)


# ─── Full pipeline — happy path ───────────────────────────────────────────────

class TestFullPipelineHappyPath:

    def test_pipeline_reaches_complete(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume, sample_critique,
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)

        # Configure mocks to return valid model instances
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        orch.critiquer.critique.return_value     = sample_critique
        orch.app_agent.assemble.return_value     = ApplicationPackage(
            job_id="test-session-001",
            cover_letter=CoverLetter(body="Dear Hiring Team...", subject_line="Re: Data Scientist"),
            resume_text=sample_rewritten_resume.full_text,
            ats_score_before=68.5,
            ats_score_after=81.0,
            keywords_added=["BigQuery"],
            ready_to_submit=True,
        )

        result = orch.run(session)

        assert result.status == JobStatus.COMPLETE
        assert result.extracted_jd is not None
        assert result.extracted_resume is not None
        assert result.analysis is not None
        assert result.rewritten_resume is not None
        assert result.package is not None
        assert result.error is None

    def test_pipeline_logs_each_phase(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume, sample_critique,
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)

        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        orch.critiquer.critique.return_value     = sample_critique
        orch.app_agent.assemble.return_value     = ApplicationPackage(
            job_id="test-session-001",
            cover_letter=CoverLetter(body="Dear...", subject_line="Re: Role"),
            resume_text="Jane Smith\nResume text",
            ready_to_submit=True,
        )

        result = orch.run(session)
        log_text = " ".join(result.agent_logs)
        assert "Phase 1" in log_text
        assert "Phase 2" in log_text
        assert "Phase 3" in log_text
        assert "Phase 4" in log_text


# ─── Phase 1 failure (critical) ───────────────────────────────────────────────

class TestScrapingFailure:

    def test_missing_resume_path_sets_failed(self, tmp_path):
        orch = _make_orch(tmp_path)
        session = JobSession(id="fail-test")
        session.resume_path = None
        session.jd_path = str(tmp_path / "jd.txt")
        (tmp_path / "jd.txt").write_text("Some JD", encoding="utf-8")

        result = orch.run(session)

        assert result.status == JobStatus.FAILED
        assert result.error is not None

    def test_empty_resume_text_sets_failed(self, tmp_path):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path, resume_text="   ", jd_text="Some JD")

        result = orch.run(session)

        assert result.status == JobStatus.FAILED
        assert any("empty" in log.lower() or "FAILED" in log for log in result.agent_logs)

    def test_scraper_exception_propagates_to_failed(self, tmp_path):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.side_effect = RuntimeError("LLM API down")

        result = orch.run(session)

        assert result.status == JobStatus.FAILED


# ─── Phase 2 failure (critical) ───────────────────────────────────────────────

class TestAnalysisFailure:

    def test_analyzer_exception_sets_failed(self, tmp_path, sample_extracted_jd, sample_extracted_resume):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.side_effect        = RuntimeError("Analysis crashed")

        result = orch.run(session)

        assert result.status == JobStatus.FAILED
        assert result.analysis is None


# ─── Phase 3 failure (non-fatal) ─────────────────────────────────────────────

class TestRewritingFailure:

    def test_rewriter_failure_falls_back_to_original(
        self, tmp_path, sample_extracted_jd, sample_extracted_resume, sample_analysis
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.side_effect        = RuntimeError("Rewriter failed")
        orch.app_agent.assemble.side_effect      = RuntimeError("Assembly skipped")

        result = orch.run(session)

        # Pipeline should still complete (not FAILED)
        assert result.status == JobStatus.COMPLETE
        # Fallback resume should match the original raw text
        assert result.rewritten_resume is not None
        assert result.rewritten_resume.full_text == sample_extracted_resume.raw_text


# ─── Phase 4 failure (non-fatal) ─────────────────────────────────────────────

class TestAssemblyFailure:

    def test_assembly_failure_still_completes(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume, sample_critique,
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        orch.critiquer.critique.return_value     = sample_critique
        orch.app_agent.assemble.side_effect      = RuntimeError("DOCX generation failed")
        # _save_resume_docx also fails (fallback)
        orch.app_agent._save_resume_docx = MagicMock(side_effect=RuntimeError("save failed"))

        result = orch.run(session)

        assert result.status == JobStatus.COMPLETE
        assert result.rewritten_resume is not None
        # Fallback package should be created
        assert result.package is not None
        assert result.package.ready_to_submit is False

    def test_fallback_package_has_cover_letter(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume, sample_critique,
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        orch.critiquer.critique.return_value     = sample_critique
        orch.app_agent.assemble.side_effect      = RuntimeError("DOCX failed")
        orch.app_agent._save_resume_docx         = MagicMock(return_value=None)

        result = orch.run(session)

        assert result.package.cover_letter is not None
        assert result.package.cover_letter.body  # non-empty


# ─── Critique loop wiring ─────────────────────────────────────────────────────

class TestCritiqueLoop:

    def test_critique_stored_in_session(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume, sample_critique,
    ):
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)
        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        orch.critiquer.critique.return_value     = sample_critique
        orch.app_agent.assemble.side_effect      = RuntimeError("skip assembly")
        orch.app_agent._save_resume_docx         = MagicMock(return_value=None)

        result = orch.run(session)

        assert result.critique is not None
        assert result.critique.passed is True

    def test_second_rewrite_triggered_when_critique_fails(
        self, tmp_path,
        sample_extracted_jd, sample_extracted_resume,
        sample_analysis, sample_rewritten_resume,
    ):
        from core.models import CritiqueIssue, CritiqueReport
        orch = _make_orch(tmp_path)
        session = _make_session(tmp_path)

        failing_critique = CritiqueReport(
            passed=False,
            overall_score=55.0,
            keyword_coverage=60.0,
            star_compliance=50.0,
            issues=[CritiqueIssue(
                section="experience",
                issue="Missing quantified result",
                severity="critical",
                suggestion="Add 'reducing costs by 30%'",
            )],
            fix_instructions="Add metrics to all bullets. Add BigQuery to skills.",
        )

        orch.scraper.extract_jd.return_value    = sample_extracted_jd
        orch.scraper.extract_resume.return_value = sample_extracted_resume
        orch.analyzer.analyze.return_value       = sample_analysis
        orch.rewriter.rewrite.return_value       = sample_rewritten_resume
        # First critique fails, second passes
        passing_critique = CritiqueReport(
            passed=True, overall_score=78.0, keyword_coverage=85.0, star_compliance=80.0
        )
        orch.critiquer.critique.side_effect = [failing_critique, passing_critique]
        orch.app_agent.assemble.side_effect  = RuntimeError("skip assembly")
        orch.app_agent._save_resume_docx     = MagicMock(return_value=None)

        result = orch.run(session)

        # rewriter.rewrite should have been called twice
        assert orch.rewriter.rewrite.call_count == 2
        # Final critique stored should be the passing one
        assert result.critique.passed is True
