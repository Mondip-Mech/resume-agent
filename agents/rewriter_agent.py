"""
agents/rewriter_agent.py
─────────────────────────
Advanced resume rewriter with:

  1. RAG bullet retrieval  — uses the candidate's own best bullets as context
  2. Multi-pass section rewriting — Summary → Experience → Skills (one at a time)
  3. Personal style injection — mirrors the user's authentic voice
  4. Critique-driven refinement — CritiqueAgent reviews each pass and directs fixes
  5. Iterative ATS loop — keeps rewriting until target score is reached

Architecture per iteration:
  Pass A: Rewrite full resume (plain text, no JSON truncation)
  Pass B: CritiqueAgent reviews → produces fix_instructions
  Pass C: Targeted fix pass using critique instructions
  → Check ATS score → repeat if needed
"""
from __future__ import annotations

import logging
import re

from core.llm_client import LLMClient
from core.memory import AgentMemory
from core.models import (
    AnalysisReport,
    CritiqueReport,
    ExtractedJD,
    ExtractedResume,
    RewrittenResume,
    RewrittenSection,
)
from tools.ats_scorer import extract_jd_keywords, score_resume

logger = logging.getLogger(__name__)


class RewriterAgent:
    """
    Master resume writer with RAG context, style learning, and self-critique.
    """

    SYSTEM_PROMPT = """You are a master resume writer with 20 years of experience.
Your ONLY goal is to make the candidate maximally qualified for the target role.

IRONCLAD RULES:
1. NEVER fabricate, invent, or embellish experience. REFRAME only what exists.
2. Use EXACT keywords from the JD — ATS requires verbatim matches, not synonyms.
3. STAR method for ALL bullets: Situation/Task → Action → Result with metric.
4. Every bullet MUST start with a strong action verb (Built, Led, Designed, etc.).
5. Add conservative estimates where implied: "improved X" → "improved X by ~25%".
6. 2 pages maximum. Cut weak content ruthlessly.
7. ATS-safe format: no tables, no columns, no special characters."""

    METADATA_SYSTEM = "You are a JSON-only responder. Return a single valid JSON object. No markdown."

    def __init__(
        self,
        llm: LLMClient,
        memory: AgentMemory = None,
        target_ats: float = 75.0,
        max_iterations: int = 2,
    ):
        self.llm = llm
        self.memory = memory or AgentMemory()
        self.target_ats = target_ats
        self.max_iterations = max_iterations

    # ─── Public API ───────────────────────────────────────────

    def rewrite(
        self,
        resume: ExtractedResume,
        jd: ExtractedJD,
        analysis: AnalysisReport,
        session_id: str = "",
        critique: CritiqueReport = None,
        style_context: str = "",
    ) -> RewrittenResume:
        """
        Full rewrite pipeline:
          1. Extract + store bullets (RAG)
          2. Build style context
          3. Multi-pass rewrite with critique feedback
          4. ATS loop until target reached
        """
        logger.info("RewriterAgent: Starting advanced rewrite pipeline...")
        jd_keywords = extract_jd_keywords(jd.raw_text, top_k=25)

        # ── Step 1: Store bullets in RAG memory ────────────────
        stored = self.memory.extract_and_store_bullets(session_id, resume.raw_text)
        logger.info(f"  Stored {len(stored)} bullets in RAG memory")

        # ── Step 2: Build style context ─────────────────────────
        if not style_context:
            style_context = self.memory.get_style_context(top_k=5)

        # ── Step 3: Get feedback insights ──────────────────────
        feedback_context = self.memory.get_feedback_insights(jd.job_title)

        # ── Step 4: Get market intelligence ────────────────────
        market_context = self.memory.get_market_summary()

        current_text = resume.raw_text
        sections: list = []
        added_keywords: list = []

        for iteration in range(self.max_iterations):
            logger.info(f"  Iteration {iteration + 1}/{self.max_iterations}")

            # ── Retrieve RAG context for this iteration ─────────
            rag_bullets = self._retrieve_rag_context(session_id, jd, analysis)

            # ── Rewrite pass (plain text — no JSON truncation) ──
            current_text = self._multi_pass_rewrite(
                resume_text=current_text,
                original_resume=resume,
                jd=jd,
                analysis=analysis,
                rag_bullets=rag_bullets,
                style_context=style_context,
                feedback_context=feedback_context,
                market_context=market_context,
                critique=critique if iteration == 0 else None,
                iteration=iteration,
            )

            # ── ATS check ───────────────────────────────────────
            new_score = score_resume(current_text, jd_keywords)
            logger.info(f"  ATS after iteration {iteration + 1}: {new_score.overall}/100")

            if new_score.overall >= self.target_ats:
                logger.info("  Target ATS reached — stopping early.")
                break

        # ── Metadata JSON (small, safe) ──────────────────────────
        meta = self._generate_metadata(resume.raw_text, current_text, jd, analysis)
        added_keywords = [str(k) for k in meta.get("keywords_added", []) if k]
        for sec_data in meta.get("sections_rewritten", []):
            if isinstance(sec_data, dict):
                sections.append(RewrittenSection(
                    section=sec_data.get("section", ""),
                    original=sec_data.get("original", ""),
                    rewritten=sec_data.get("rewritten", ""),
                    changes_made=sec_data.get("changes_made", []),
                ))

        # ── Last-resort keyword injection ────────────────────────
        # If critical JD keywords are still absent after all LLM passes,
        # inject them directly into the Skills section so ATS always sees them.
        final_ats = score_resume(current_text, jd_keywords)
        critical_gaps = [
            kw for kw in final_ats.missing_keywords
            if any(
                kw.lower() == g.requirement.lower()
                for g in analysis.skill_gaps
                if g.importance == "required"
            )
        ]
        if critical_gaps:
            logger.info(
                f"  Force-injecting {len(critical_gaps)} still-missing critical keywords: "
                f"{critical_gaps}"
            )
            current_text = self._force_keywords_into_skills(current_text, critical_gaps)
            final_ats = score_resume(current_text, jd_keywords)

        logger.info(f"RewriterAgent: Final ATS = {final_ats.overall}/100")

        return RewrittenResume(
            job_id=session_id,
            full_text=current_text,
            sections=sections,
            added_keywords=added_keywords,
            removed_content=meta.get("removed_content", []),
            new_ats_score=final_ats,
        )

    # ─── RAG Context Retrieval ────────────────────────────────

    def _retrieve_rag_context(
        self, session_id: str, jd: ExtractedJD, analysis: AnalysisReport
    ) -> str:
        """
        For each required JD skill, retrieve the most relevant bullet from
        the candidate's actual resume to use as evidence for the rewriter.
        """
        rag_lines = []
        key_requirements = [
            g.requirement for g in analysis.skill_gaps
            if g.importance == "required"
        ][:6]

        for req in key_requirements:
            hits = self.memory.retrieve_relevant_bullets(session_id, req, top_k=2)
            for hit in hits:
                if hit and hit not in rag_lines:
                    rag_lines.append(f"  [{req[:40]}] → \"{hit}\"")

        if not rag_lines:
            return ""

        return (
            "EVIDENCE FROM CANDIDATE'S ACTUAL RESUME (use these verbatim, reframe to match JD):\n"
            + "\n".join(rag_lines)
        )

    # ─── Multi-Pass Rewrite ────────────────────────────────────

    def _multi_pass_rewrite(
        self,
        resume_text: str,
        original_resume: ExtractedResume,
        jd: ExtractedJD,
        analysis: AnalysisReport,
        rag_bullets: str,
        style_context: str,
        feedback_context: str,
        market_context: str,
        critique: CritiqueReport = None,
        iteration: int = 0,
    ) -> str:
        """
        Section-aware rewrite:
          1. Full resume rewrite with all context
          2. If critique provided: targeted fix pass
        """
        # ── Full rewrite pass ───────────────────────────────────
        full_text = self._full_rewrite(
            resume_text=resume_text,
            jd=jd,
            analysis=analysis,
            rag_bullets=rag_bullets,
            style_context=style_context,
            feedback_context=feedback_context,
            market_context=market_context,
            iteration=iteration,
        )

        if not full_text or len(full_text) < 200:
            logger.warning("  Full rewrite returned empty — keeping previous text.")
            return resume_text

        # ── Targeted fix pass (if critique provided) ───────────
        if critique and not critique.passed and critique.fix_instructions:
            logger.info("  Applying critique-driven targeted fixes...")
            fixed = self._critique_fix_pass(
                rewritten_text=full_text,
                jd=jd,
                fix_instructions=critique.fix_instructions,
                critical_issues=[
                    i for i in critique.issues if i.severity == "critical"
                ],
            )
            if fixed and len(fixed) > 200:
                return fixed

        return full_text

    def _full_rewrite(
        self,
        resume_text: str,
        jd: ExtractedJD,
        analysis: AnalysisReport,
        rag_bullets: str,
        style_context: str,
        feedback_context: str,
        market_context: str,
        iteration: int,
    ) -> str:
        """Main full-resume rewrite with all context injected."""
        priority_rewrites = "\n".join(analysis.priority_rewrites[:8])
        missing_keywords = ", ".join(analysis.ats_score.missing_keywords[:15])
        gaps_summary = "\n".join([
            f"- {g.requirement}: {g.rewrite_suggestion or 'add to resume'}"
            for g in analysis.skill_gaps
            if g.match.value in ("missing", "partial") and g.importance == "required"
        ][:8])

        extra_context = "\n\n".join(filter(None, [
            rag_bullets,
            style_context,
            feedback_context,
            market_context,
        ]))

        iteration_note = (
            f"\nThis is REFINEMENT PASS {iteration + 1}. The previous pass scored below target.\n"
            f"MUST FIX:\n"
            f"1. Add ALL of these keywords verbatim to the Skills section: {missing_keywords[:200]}\n"
            f"2. Every bullet must end with a quantified result (%, $, x improvement).\n"
            f"3. Summary first line must start with the exact job title: {jd.job_title}."
            if iteration > 0 else ""
        )

        prompt = f"""Rewrite this resume to maximally target: {jd.job_title} at {jd.company}.
{iteration_note}

ORIGINAL RESUME:
{resume_text[:4500]}

{extra_context}

STRATEGY (from gap analysis):
{priority_rewrites}

CRITICAL GAPS TO ADDRESS:
{gaps_summary}

KEYWORDS TO ADD VERBATIM (ATS critical): {missing_keywords}

JD TONE/CULTURE SIGNALS: {', '.join(jd.culture_signals[:5])}

SECTION-BY-SECTION INSTRUCTIONS:
1. SUMMARY (3-4 lines): Start with target job title. Mirror JD language. Quantify impact.
2. SKILLS: List all required JD keywords explicitly. Group by category.
3. EXPERIENCE (each role): Rewrite every bullet with STAR format + metric.
   - Start each bullet with an action verb
   - Add quantified results (use ~estimates if not stated)
   - Inject verbatim JD keywords naturally
4. PROJECTS: Highlight technical depth and measurable outcomes.
5. EDUCATION/CERTS: Keep factual. Add any relevant courses.

OUTPUT: Return ONLY the complete rewritten resume as plain text.
- Do NOT use JSON, markdown, or code fences.
- Do NOT add commentary before or after.
- Start with the candidate's name on line 1."""

        raw = self.llm.call(prompt, system=self.SYSTEM_PROMPT, max_tokens=2048)

        # Handle case where LLM returns JSON despite instruction
        if raw and raw.strip().startswith("{") and '"full_resume_text"' in raw:
            try:
                import json as _json
                parsed = _json.loads(re.sub(r"```(?:json)?\s*", "", raw).strip())
                extracted = parsed.get("full_resume_text", "")
                if extracted and len(extracted) > 200:
                    return extracted
            except Exception:
                pass
        return raw

    def _critique_fix_pass(
        self,
        rewritten_text: str,
        jd: ExtractedJD,
        fix_instructions: str,
        critical_issues: list,
    ) -> str:
        """Targeted fix pass based on specific critique instructions."""
        issues_text = "\n".join(
            f"- [{i.section}] {i.issue} → FIX: {i.suggestion}"
            for i in critical_issues[:5]
        )

        prompt = f"""Apply these specific fixes to the resume for role: {jd.job_title}

CRITIQUE INSTRUCTIONS:
{fix_instructions}

SPECIFIC ISSUES TO FIX:
{issues_text or 'None — apply general improvements from instructions above.'}

CURRENT RESUME:
{rewritten_text[:4500]}

Apply ALL the fixes above. Return ONLY the corrected resume as plain text.
Keep everything else unchanged. Start with the candidate's name."""

        raw = self.llm.call(prompt, system=self.SYSTEM_PROMPT, max_tokens=2048)
        return raw if raw and len(raw) > 200 else rewritten_text

    # ─── Force keyword injection ──────────────────────────────

    def _force_keywords_into_skills(self, resume_text: str, keywords: list) -> str:
        """
        Append still-missing keywords to the Skills section.
        Falls back to appending a new Skills block if no section is found.
        This guarantees ATS keyword presence regardless of LLM behaviour.
        """
        addition = ", ".join(keywords)

        # Try to find and extend an existing Skills section
        skills_re = re.compile(
            r"(SKILLS?[^\n]*\n)(.*?)(\n{2,}|\n(?=[A-Z][A-Z\s]{2,}:?\n))",
            re.DOTALL | re.IGNORECASE,
        )
        m = skills_re.search(resume_text)
        if m:
            header, body, tail = m.group(1), m.group(2), m.group(3)
            new_body = body.rstrip(", \n") + f", {addition}"
            return (
                resume_text[: m.start()]
                + header + new_body + tail
                + resume_text[m.end():]
            )

        # No Skills section — insert one before EXPERIENCE
        exp_re = re.compile(r"\n(EXPERIENCE|WORK EXPERIENCE|PROFESSIONAL EXPERIENCE)", re.IGNORECASE)
        exp_m = exp_re.search(resume_text)
        block = f"\nSKILLS\n{addition}\n"
        if exp_m:
            return resume_text[: exp_m.start()] + block + resume_text[exp_m.start():]
        return resume_text + block

    # ─── Metadata JSON ────────────────────────────────────────

    def _generate_metadata(
        self,
        original_text: str,
        rewritten_text: str,
        jd: ExtractedJD,
        analysis: AnalysisReport,
    ) -> dict:
        """Small JSON call for keywords/sections metadata only."""
        prompt = f"""Compare original and rewritten resume for: {jd.job_title}

ORIGINAL (first 600 chars): {original_text[:600]}
REWRITTEN (first 600 chars): {rewritten_text[:600]}

Return JSON:
{{
  "keywords_added": ["kw1", "kw2"],
  "removed_content": ["item1"],
  "sections_rewritten": [
    {{"section": "summary", "original": "...", "rewritten": "...", "changes_made": ["change1"]}}
  ]
}}"""
        try:
            return self.llm.call_json(prompt, system=self.METADATA_SYSTEM, max_tokens=800)
        except Exception as e:
            logger.warning(f"Metadata call failed: {e}")
            return {"keywords_added": [], "removed_content": [], "sections_rewritten": []}
