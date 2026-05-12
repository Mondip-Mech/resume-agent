"""
agents/analyzer_agent.py
─────────────────────────
Performs deep gap analysis between JD requirements and resume content.

Uses Chain-of-Thought reasoning to:
  1. Compare every JD requirement against resume evidence
  2. Score match strength (strong / partial / missing)
  3. Generate specific rewriting strategies for each gap
  4. Compute ATS score before any rewriting
  5. Prioritize rewrites by importance and impact
"""
from __future__ import annotations

import logging
from typing import List

from core.llm_client import LLMClient
from core.models import (
    AnalysisReport,
    ExtractedJD,
    ExtractedResume,
    MatchStrength,
    SeniorityLevel,
    SkillGap,
)
from tools.ats_scorer import extract_jd_keywords, score_resume

logger = logging.getLogger(__name__)


class AnalyzerAgent:
    """
    Persona: Ruthless but constructive career strategist.
    Key output: AnalysisReport with actionable, evidence-based strategies.
    Uses CoT — always shows comparison before making recommendations.
    """

    SYSTEM_PROMPT = """You are the Resume Strategist — a ruthless but constructive career coach and ATS expert.

CRITICAL RULES:
1. Never suggest fabricating experience. Only suggest reframing existing experience.
2. Always show the comparison BEFORE making a suggestion (Chain-of-Thought).
3. Be specific: "Rewrite bullet 3 to mention Python explicitly" NOT "Improve skills section".
4. Prioritize: required requirements before preferred before nice_to_have.
5. Output must be valid JSON."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    def analyze(self, jd: ExtractedJD, resume: ExtractedResume, session_id: str = "") -> AnalysisReport:
        """
        Full gap analysis with ATS scoring.

        Returns AnalysisReport with:
          - skill_gaps (each with a specific rewrite suggestion)
          - ATS score before rewriting
          - prioritized rewrite strategy
          - CoT reasoning notes
        """
        logger.info("AnalyzerAgent: Running gap analysis (Chain-of-Thought)...")

        # Step 1: Extract keywords for ATS scoring
        jd_keywords = extract_jd_keywords(jd.raw_text, top_k=25)

        # Step 2: Compute current ATS score
        ats_before = score_resume(resume.raw_text, jd_keywords)
        logger.info(f"  ATS score (before): {ats_before.overall}/100")

        # Step 3: LLM gap analysis with CoT
        gap_data = self._run_gap_analysis(jd, resume)

        # Step 4: Seniority match check
        seniority_match = self._check_seniority_match(jd.seniority, resume)

        # Step 5: Build skill gaps
        skill_gaps = self._parse_skill_gaps(gap_data)

        # Step 6: Overall match score (weighted keyword + gap)
        missing_required = sum(
            1 for g in skill_gaps
            if g.match == MatchStrength.MISSING and g.importance == "required"
        )
        match_score = max(0, ats_before.overall - (missing_required * 8))

        def _to_str_list(v):
            if not isinstance(v, list):
                return []
            return [str(item) if not isinstance(item, dict)
                    else str(next(iter(item.values()), ""))
                    for item in v if item is not None]

        return AnalysisReport(
            job_id=session_id,
            match_score=round(match_score, 1),
            ats_score=ats_before,
            seniority_match=seniority_match,
            skill_gaps=skill_gaps,
            strengths=_to_str_list(gap_data.get("strengths", [])),
            priority_rewrites=_to_str_list(gap_data.get("priority_rewrites", [])),
            strategy_notes=str(gap_data.get("strategy_notes", "") or ""),
        )

    # ─── Private helpers ──────────────────────────────────────

    def _run_gap_analysis(self, jd: ExtractedJD, resume: ExtractedResume) -> dict:
        """Core LLM call for gap analysis with CoT."""
        req_summary = "\n".join([
            f"- [{r.importance.upper()}] {r.text} (keywords: {', '.join(r.keywords[:3])})"
            for r in jd.requirements[:30]
        ])

        experience_summary = ""
        for exp in resume.experience[:5]:
            bullets = "\n    ".join(exp.get("bullets", [])[:5])
            experience_summary += (
                f"\n  {exp.get('title')} at {exp.get('company')} ({exp.get('dates')})\n"
                f"    {bullets}\n"
            )

        prompt = f"""You are the Resume Strategist. Analyze the fit between this job description and resume.

JOB: {jd.job_title} at {jd.company}
REQUIREMENTS:
{req_summary}

CANDIDATE RESUME:
Current Title: {resume.current_title}
Skills: {', '.join(resume.skills[:20])}
Experience:
{experience_summary}

CHAIN-OF-THOUGHT INSTRUCTIONS:
For each requirement, think through:
1. Is this skill/experience present in the resume?
2. Is it explicitly stated or only implied?
3. What is the match strength: strong / partial / missing?
4. If not strong: what SPECIFIC rewrite would fix this?
   (Quote the exact bullet to rewrite, and what to add)

Return this JSON:
{{
  "gaps": [
    {{
      "requirement": "exact JD requirement text",
      "category": "skill|experience|education|soft_skill",
      "importance": "required|preferred|nice_to_have",
      "match": "strong|partial|missing",
      "resume_evidence": "the specific resume text that proves this (or null)",
      "cot_reasoning": "I checked... and found... therefore...",
      "rewrite_suggestion": "Specific: rewrite bullet X to say '...' to mention keyword Y"
    }}
  ],
  "strengths": ["Resume clearly demonstrates X", "Strong evidence of Y from Z role"],
  "priority_rewrites": [
    "1. [REQUIRED] Add Python to skills section — appears 5× in JD",
    "2. [REQUIRED] Rewrite job 1 bullet 3 to quantify the performance improvement",
    "3. [PREFERRED] Add mention of Agile/Scrum methodology to job 2"
  ],
  "strategy_notes": "Overall: The candidate has 70% of required skills but several are only implied. Focus on: 1) Making Python explicit, 2) Adding metrics to 3 bullets, 3) Rewriting summary to echo the JD's language about 'scalable systems'."
}}"""

        return self.llm.call_json(prompt, system=self.SYSTEM_PROMPT, max_tokens=2048)

    def _parse_skill_gaps(self, data: dict) -> List[SkillGap]:
        gaps = []
        for g in data.get("gaps", []):
            try:
                match = MatchStrength(g.get("match", "missing"))
            except ValueError:
                match = MatchStrength.MISSING
            gaps.append(SkillGap(
                requirement=g.get("requirement", ""),
                category=g.get("category", "skill"),
                importance=g.get("importance", "preferred"),
                match=match,
                resume_evidence=g.get("resume_evidence"),
                rewrite_suggestion=g.get("rewrite_suggestion"),
            ))
        return gaps

    def _check_seniority_match(self, jd_seniority: SeniorityLevel, resume: ExtractedResume) -> bool:
        """Rough seniority match — count years of experience."""
        if jd_seniority == SeniorityLevel.UNKNOWN:
            return True

        import re
        from datetime import datetime
        current_year = datetime.utcnow().year
        years = 0
        for exp in resume.experience:
            dates = exp.get("dates", "")
            # Replace "Present"/"Current"/"Now" with the current year so the
            # regex finds two year numbers and computes the correct duration.
            dates_normalized = re.sub(
                r"\b(?:present|current|now)\b", str(current_year), dates, flags=re.IGNORECASE
            )
            year_matches = re.findall(r"\b(20\d{2}|19\d{2})\b", dates_normalized)
            if len(year_matches) >= 2:
                try:
                    years += abs(int(year_matches[-1]) - int(year_matches[0]))
                except ValueError:
                    pass

        thresholds = {
            SeniorityLevel.JUNIOR:    (0, 3),
            SeniorityLevel.MID:       (2, 6),
            SeniorityLevel.SENIOR:    (5, 99),
            SeniorityLevel.LEAD:      (7, 99),
            SeniorityLevel.PRINCIPAL: (10, 99),
            SeniorityLevel.EXECUTIVE: (12, 99),
        }
        low, high = thresholds.get(jd_seniority, (0, 99))
        return low <= years <= high
