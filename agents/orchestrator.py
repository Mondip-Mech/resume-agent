"""
agents/orchestrator.py
───────────────────────
The central Planner-Executor Orchestrator.

Responsibilities:
  1. Manage the full pipeline state machine
  2. Route between agents in correct order
  3. Validate each agent's output before passing to the next
  4. Handle errors with retry and fallback strategies
  5. Maintain the JobSession state throughout
  6. Produce a final audit summary

State machine:
  PENDING → SCRAPING → ANALYZING → REWRITING → ASSEMBLING → COMPLETE
                                                              ↓ (on error)
                                                            FAILED
"""
from __future__ import annotations

import logging
import traceback

from agents.analyzer_agent import AnalyzerAgent
from agents.application_agent import ApplicationAgent
from agents.critique_agent import CritiqueAgent
from agents.rewriter_agent import RewriterAgent
from agents.scraper_agent import ScraperAgent
from core.config import get_settings
from core.llm_client import LLMClient
from core.memory import AgentMemory
from core.models import JobSession, JobStatus
from parsers.pdf_parser import extract_text

logger = logging.getLogger(__name__)


class Orchestrator:
    """
    Planner-Executor that manages the full job application pipeline.

    Usage:
        orchestrator = Orchestrator()
        session = orchestrator.run(session)
    """

    def __init__(self):
        cfg = get_settings()
        self.llm = LLMClient(
            api_key=cfg.nvidia_api_key,
            model=cfg.llm_model,
            max_tokens=cfg.llm_max_tokens,
            temperature=cfg.llm_temperature,
        )
        self.memory = AgentMemory(persist_dir=cfg.chroma_dir)

        # Initialize agents
        self.scraper    = ScraperAgent(self.llm)
        self.analyzer   = AnalyzerAgent(self.llm)
        self.critiquer  = CritiqueAgent(self.llm, pass_threshold=cfg.target_ats_score - 3)
        self.rewriter   = RewriterAgent(
            self.llm,
            memory=self.memory,
            target_ats=cfg.target_ats_score,
            max_iterations=cfg.rewrite_iterations,
        )
        self.app_agent  = ApplicationAgent(self.llm, output_dir=cfg.output_dir)

    # ─── Public API ───────────────────────────────────────────

    def run(self, session: JobSession) -> JobSession:
        """
        Execute the full pipeline synchronously.
        Each phase is run independently — a later phase failure never
        discards results already produced by earlier phases.
        """
        logger.info(f"Orchestrator: Starting pipeline for session {session.id}")
        session.log("Pipeline started.")

        # ── Phase 1: Scraping (critical — must succeed) ───────────────────────
        try:
            session = self._phase_scraping(session)
        except Exception as e:
            session.status = JobStatus.FAILED
            session.error = str(e)
            session.log(f"FAILED at scraping: {e}\n{traceback.format_exc()}")
            logger.error(f"Scraping failed for {session.id}: {e}")
            return session

        # ── Phase 2: Analysis (critical — must succeed) ───────────────────────
        try:
            session = self._phase_analyzing(session)
        except Exception as e:
            session.status = JobStatus.FAILED
            session.error = str(e)
            session.log(f"FAILED at analysis: {e}\n{traceback.format_exc()}")
            logger.error(f"Analysis failed for {session.id}: {e}")
            return session

        # ── Phase 3: Rewriting (important — keep original if fails) ──────────
        try:
            session = self._phase_rewriting(session)
        except Exception as e:
            session.log(
                f"WARNING: Resume rewriting failed ({e}). "
                "Using original resume text as fallback."
            )
            logger.warning(f"Rewriting failed for {session.id}: {e}")
            # Build a minimal rewritten resume from the original text
            from core.models import RewrittenResume
            from tools.ats_scorer import extract_jd_keywords, score_resume
            jd_kw = extract_jd_keywords(session.extracted_jd.raw_text, top_k=25)
            session.rewritten_resume = RewrittenResume(
                job_id=session.id,
                full_text=session.extracted_resume.raw_text,
                new_ats_score=score_resume(session.extracted_resume.raw_text, jd_kw),
            )

        # ── Phase 4: Assembling (non-fatal — covered by _phase_assembling) ───
        try:
            session = self._phase_assembling(session)
        except Exception as e:
            # Should not reach here — _phase_assembling has its own try/except,
            # but guard just in case.
            session.log(f"WARNING: Assembly failed ({e}). Resume still available.")
            logger.warning(f"Assembly failed for {session.id}: {e}")

        session.status = JobStatus.COMPLETE
        session.log("Pipeline complete.")
        logger.info(f"Pipeline complete. Token usage: {self.llm.token_usage}")

        return session

    # ─── Phase 1: Scraping ────────────────────────────────────

    def _phase_scraping(self, session: JobSession) -> JobSession:
        session.status = JobStatus.SCRAPING
        session.log("Phase 1: Scraping and parsing documents...")

        # Extract raw text from files
        resume_text = self._load_document(session.resume_path, "resume")
        jd_text = self._load_document(session.jd_path, "job description")

        if not resume_text.strip():
            raise ValueError("Resume text is empty. Check the file format.")
        if not jd_text.strip():
            raise ValueError("Job description text is empty. Check the file format.")

        # Store in vector memory for semantic search + bullet-level RAG
        self.memory.store_resume(session.id, resume_text)
        self.memory.store_jd(session.id, jd_text)
        self.memory.store_job_posting(jd_text)   # market intelligence
        session.log("Documents stored in vector memory.")

        # LLM extraction
        session.extracted_jd = self.scraper.extract_jd(jd_text)
        session.log(
            f"JD extracted: {session.extracted_jd.job_title} at {session.extracted_jd.company} "
            f"({len(session.extracted_jd.requirements)} requirements)"
        )

        session.extracted_resume = self.scraper.extract_resume(resume_text)
        session.log(
            f"Resume extracted: {session.extracted_resume.name}, "
            f"{len(session.extracted_resume.skills)} skills found."
        )

        # Validate outputs
        self._validate_scraper_output(session)

        return session

    # ─── Phase 2: Analysis ────────────────────────────────────

    def _phase_analyzing(self, session: JobSession) -> JobSession:
        session.status = JobStatus.ANALYZING
        session.log("Phase 2: Running gap analysis (Chain-of-Thought)...")

        session.analysis = self.analyzer.analyze(
            jd=session.extracted_jd,
            resume=session.extracted_resume,
            session_id=session.id,
        )

        ats = session.analysis.ats_score
        session.log(
            f"Analysis complete. Match score: {session.analysis.match_score}/100. "
            f"ATS: {ats.overall}/100 "
            f"(keywords: {ats.keyword_match:.0f}%, "
            f"quant: {ats.quantification:.0f}%). "
            f"{len(session.analysis.skill_gaps)} gaps found."
        )
        missing_required_skills = [
            g.requirement for g in session.analysis.skill_gaps
            if g.match.value == "missing" and g.importance == "required"
        ]
        session.log(
            "Missing required skills: " + (", ".join(missing_required_skills[:3]) or "none")
        )

        return session

    # ─── Phase 3: Rewriting ───────────────────────────────────

    def _phase_rewriting(self, session: JobSession) -> JobSession:
        session.status = JobStatus.REWRITING
        session.log("Phase 3: Rewriting and tailoring resume...")

        # Pull personalisation context from memory
        style_context = self.memory.get_style_context(top_k=5)

        # ── Initial rewrite ────────────────────────────────────────────────────
        session.rewritten_resume = self.rewriter.rewrite(
            resume=session.extracted_resume,
            jd=session.extracted_jd,
            analysis=session.analysis,
            session_id=session.id,
            style_context=style_context,
        )

        new_ats = session.rewritten_resume.new_ats_score
        old_ats = session.analysis.ats_score.overall
        improvement = (new_ats.overall - old_ats) if new_ats else 0

        session.log(
            f"Rewrite complete. ATS: {old_ats} → {new_ats.overall if new_ats else '?'} "
            f"({'+' if improvement >= 0 else ''}{improvement:.1f}pts). "
            f"Keywords added: {', '.join(session.rewritten_resume.added_keywords[:5])}."
        )

        # ── Self-Critique pass ─────────────────────────────────────────────────
        try:
            session.critique = self.critiquer.critique(
                original_resume=session.extracted_resume,
                rewritten_text=session.rewritten_resume.full_text,
                jd=session.extracted_jd,
                analysis=session.analysis,
            )
            session.log(
                f"Critique: score={session.critique.overall_score:.0f}/100  "
                f"passed={session.critique.passed}  "
                f"{len(session.critique.issues)} issue(s) found."
            )

            # ── Targeted fix pass if critique did not pass ─────────────────────
            if not session.critique.passed and session.critique.fix_instructions:
                session.log("Critique not passed — running targeted fix pass...")
                session.rewritten_resume = self.rewriter.rewrite(
                    resume=session.extracted_resume,
                    jd=session.extracted_jd,
                    analysis=session.analysis,
                    session_id=session.id,
                    critique=session.critique,
                    style_context=style_context,
                )
                final_ats = session.rewritten_resume.new_ats_score
                session.log(
                    f"Post-critique rewrite complete. "
                    f"ATS: {final_ats.overall if final_ats else '?'}/100."
                )

                # ── Re-run critique on the fixed resume so UI shows final state ─
                try:
                    updated_critique = self.critiquer.critique(
                        original_resume=session.extracted_resume,
                        rewritten_text=session.rewritten_resume.full_text,
                        jd=session.extracted_jd,
                        analysis=session.analysis,
                    )
                    session.critique = updated_critique   # replace with final verdict
                    session.log(
                        f"Final critique: score={updated_critique.overall_score:.0f}/100  "
                        f"passed={updated_critique.passed}  "
                        f"{len(updated_critique.issues)} issue(s) remaining."
                    )
                except Exception as e:
                    session.log(f"WARNING: Post-fix critique failed ({e}). Keeping first critique.")
        except Exception as e:
            session.log(f"WARNING: Critique step failed ({e}). Keeping initial rewrite.")
            logger.warning(f"Critique failed for {session.id}: {e}")

        # Validate: ensure content was not lost
        self._validate_rewrite(session)

        return session

    # ─── Phase 4: Assembling ──────────────────────────────────

    def _phase_assembling(self, session: JobSession) -> JobSession:
        session.status = JobStatus.ASSEMBLING
        session.log("Phase 4: Assembling application package...")

        try:
            session.package = self.app_agent.assemble(
                jd=session.extracted_jd,
                resume=session.extracted_resume,
                rewritten=session.rewritten_resume,
                analysis=session.analysis,
                session_id=session.id,
            )
            session.log(
                f"Package assembled. Ready: {session.package.ready_to_submit}. "
                f"Files: resume={session.package.resume_docx_path}, "
                f"cover_letter={session.package.cover_letter_docx_path}."
            )
            if not session.package.ready_to_submit:
                session.log(f"Submission notes: {session.package.submission_notes}")

        except Exception as e:
            # Phase 4 failure is non-fatal — the rewritten resume is already done.
            # Build a minimal package so the UI can still show and download the resume.
            logger.warning(f"Phase 4 partial failure: {e}. Building fallback package.")
            session.log(
                f"WARNING: Cover letter generation failed ({e}). "
                "Resume is complete and downloadable. Please write the cover letter manually."
            )
            session.package = self._build_fallback_package(session)

        return session

    def _build_fallback_package(self, session: JobSession):
        """Return a minimal ApplicationPackage when cover-letter generation fails."""
        from core.models import ApplicationPackage, CoverLetter

        jd   = session.extracted_jd
        res  = session.extracted_resume
        rwr  = session.rewritten_resume
        ana  = session.analysis

        # Try to save just the resume file so the download button works
        resume_path = None
        try:
            resume_path = self.app_agent._save_resume_docx(rwr, session.id)
        except Exception as save_err:
            logger.warning(f"Fallback resume save also failed: {save_err}")

        return ApplicationPackage(
            job_id=session.id,
            cover_letter=CoverLetter(
                body=(
                    f"Dear Hiring Team at {jd.company},\n\n"
                    f"Please find my tailored resume for the {jd.job_title} position. "
                    "I am excited about this opportunity and believe my background "
                    "aligns well with your requirements.\n\n"
                    "[Please personalise this cover letter before submitting.]\n\n"
                    "I would welcome a conversation to discuss how I can contribute.\n\n"
                    "Best regards,\n"
                    f"{res.name}"
                ),
                subject_line=f"Re: {jd.job_title} — {res.name}",
                tone="professional",
            ),
            resume_text=rwr.full_text,
            resume_docx_path=str(resume_path) if resume_path else None,
            ats_score_before=ana.ats_score.overall if ana else None,
            ats_score_after=rwr.new_ats_score.overall if rwr.new_ats_score else None,
            keywords_added=rwr.added_keywords,
            ready_to_submit=False,
            submission_notes="Cover letter generation failed — please write manually.",
        )

    # ─── Helpers ──────────────────────────────────────────────

    def _load_document(self, path: str | None, name: str) -> str:
        if not path:
            raise ValueError(f"No {name} file provided.")
        try:
            return extract_text(path)
        except Exception as e:
            raise ValueError(f"Failed to read {name} from '{path}': {e}")

    def _validate_scraper_output(self, session: JobSession):
        """Sanity checks on extracted data."""
        jd = session.extracted_jd
        resume = session.extracted_resume

        if not jd.requirements:
            session.log("WARNING: No requirements extracted from JD. Continuing.")
        if not resume.experience:
            session.log("WARNING: No work experience found in resume. Verify file quality.")
        if not resume.skills:
            session.log("WARNING: No skills extracted from resume.")

    def _validate_rewrite(self, session: JobSession):
        """Ensure rewrite didn't lose important information."""
        original_name = session.extracted_resume.name.lower()
        rewritten_text = session.rewritten_resume.full_text.lower()

        if original_name and original_name not in rewritten_text:
            session.log(
                f"WARNING: Candidate name '{session.extracted_resume.name}' "
                "not found in rewritten resume. May need manual correction."
            )

        if len(session.rewritten_resume.full_text) < 300:
            session.log("WARNING: Rewritten resume seems very short. Check output quality.")
