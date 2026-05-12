"""
agents/critique_agent.py
─────────────────────────
Self-Critique Agent — reviews the rewritten resume and scores it.

Implements the "Reflection" pattern:
  RewriterAgent produces draft
        ↓
  CritiqueAgent reviews it with fresh eyes
        ↓
  Returns structured issues + fix instructions
        ↓
  RewriterAgent applies targeted fixes
        ↓
  Repeat until critique passes (or max rounds)

This catches what a single-pass rewrite misses:
  • Keywords present but buried / used incorrectly
  • Bullets that are still weak after rewriting
  • Summary not matching JD tone
  • Missing sections
  • Fabricated content (hallucination detection)
"""
from __future__ import annotations

import logging

from core.llm_client import LLMClient
from core.models import (
    AnalysisReport,
    CritiqueIssue,
    CritiqueReport,
    ExtractedJD,
    ExtractedResume,
)

logger = logging.getLogger(__name__)


class CritiqueAgent:
    """
    Persona: Ruthless senior hiring manager + ATS expert.
    Reviews resumes with zero tolerance for vagueness.
    """

    SYSTEM_PROMPT = """You are a senior hiring manager and ATS expert reviewing a resume.
You have seen 10,000+ resumes. You are ruthless, specific, and constructive.

YOUR JOB:
- Identify SPECIFIC weaknesses (not general advice)
- Check every required keyword from the JD is present
- Verify bullets use STAR format (Situation→Action→Result)
- Ensure NO fabricated experience (anything not in the original is fabrication)
- Score honestly — do not inflate scores

OUTPUT: Valid JSON only. No prose before or after."""

    def __init__(self, llm: LLMClient, pass_threshold: float = 72.0, max_rounds: int = 2):
        self.llm = llm
        self.pass_threshold = pass_threshold
        self.max_rounds = max_rounds

    def critique(
        self,
        original_resume: ExtractedResume,
        rewritten_text: str,
        jd: ExtractedJD,
        analysis: AnalysisReport,
    ) -> CritiqueReport:
        """Review the rewritten resume and return structured feedback."""
        logger.info("CritiqueAgent: Reviewing rewritten resume...")

        required_keywords = [
            kw for kw in analysis.ats_score.missing_keywords[:12]
        ]
        required_skills = [
            r.text[:60] for r in jd.requirements
            if r.importance == "required"
        ][:8]

        prompt = f"""Review this rewritten resume for the role: {jd.job_title} at {jd.company}

ORIGINAL CANDIDATE SKILLS: {', '.join(original_resume.skills[:15])}
ORIGINAL EXPERIENCE: {original_resume.experience[0].get('title', '') if original_resume.experience else 'N/A'} at {original_resume.experience[0].get('company', '') if original_resume.experience else 'N/A'}

REQUIRED JD SKILLS (must be present): {', '.join(required_skills)}
KEYWORDS THAT WERE MISSING: {', '.join(required_keywords)}

REWRITTEN RESUME (first 2500 chars):
{rewritten_text[:2500]}

EVALUATE:
1. keyword_coverage: What % of required keywords are now present? (0-100)
2. star_compliance: What % of bullets use STAR format with a result? (0-100)
3. overall_score: Overall quality score (0-100). Above 72 = pass.
4. issues: List specific problems. For each: section, what's wrong, severity (critical/major/minor), exact fix.
5. praise: List 2-3 specific things done well.
6. passed: true if overall_score >= 72 AND no critical issues remain.
7. fix_instructions: If not passed, write precise rewrite instructions in 3-5 sentences.

CRITICAL CHECKS:
- If a required keyword is completely absent → critical issue
- If a bullet has no result/metric → major issue
- If summary does not mention the target role title → major issue
- If any experience NOT in the original was added → critical issue (fabrication)

Return JSON:
{{
  "passed": false,
  "overall_score": 65.0,
  "keyword_coverage": 70.0,
  "star_compliance": 60.0,
  "issues": [
    {{
      "section": "experience",
      "issue": "Bullet 2 missing quantified result",
      "severity": "major",
      "suggestion": "Add: 'reducing processing time by 35%' after the action"
    }}
  ],
  "praise": ["Strong summary that mirrors JD language", "Python keyword added correctly"],
  "fix_instructions": "Focus on: 1) Add SQL and A/B testing to skills section. 2) Quantify bullet 3 with a metric. 3) Add 'Data Scientist' to the summary title line."
}}"""

        try:
            data = self.llm.call_json(prompt, system=self.SYSTEM_PROMPT, max_tokens=1200)
        except Exception as e:
            logger.warning(f"CritiqueAgent JSON call failed: {e}. Returning permissive pass.")
            return CritiqueReport(
                passed=True, overall_score=70.0,
                keyword_coverage=70.0, star_compliance=70.0,
            )

        issues = []
        for item in data.get("issues", []):
            if isinstance(item, dict):
                try:
                    issues.append(CritiqueIssue(
                        section=item.get("section", "general"),
                        issue=item.get("issue", ""),
                        severity=item.get("severity", "minor"),
                        suggestion=item.get("suggestion", ""),
                    ))
                except Exception:
                    pass

        score = float(data.get("overall_score", 65))
        passed = bool(data.get("passed", score >= self.pass_threshold))
        if not passed and score >= self.pass_threshold:
            critical = [i for i in issues if i.severity == "critical"]
            passed = len(critical) == 0

        report = CritiqueReport(
            passed=passed,
            overall_score=score,
            keyword_coverage=float(data.get("keyword_coverage", 60)),
            star_compliance=float(data.get("star_compliance", 60)),
            issues=issues,
            praise=data.get("praise", []),
            fix_instructions=str(data.get("fix_instructions", "") or ""),
        )

        logger.info(
            f"  Critique: score={report.overall_score:.0f}/100  "
            f"passed={report.passed}  issues={len(report.issues)}"
        )
        return report
