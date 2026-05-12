"""
agents/scraper_agent.py
────────────────────────
Extracts structured data from raw JD and Resume text.

Output:
  • ExtractedJD  — structured job requirements
  • ExtractedResume — parsed resume with skill inventory
"""
from __future__ import annotations

import logging

from core.llm_client import LLMClient
from core.models import (
    ExtractedJD,
    ExtractedResume,
    JDRequirement,
    ResumeSkill,
    SeniorityLevel,
)

logger = logging.getLogger(__name__)


class ScraperAgent:
    """
    Persona: Precise data extraction specialist.
    Hallucination risk: Low (extraction only, no synthesis).
    """

    SYSTEM_PROMPT = """You are a precise Job Description and Resume Analyst.
Extract ONLY information explicitly stated. Never infer, add, or hallucinate.
Always output valid JSON matching the requested structure exactly."""

    def __init__(self, llm: LLMClient):
        self.llm = llm

    # ─── JD Extraction ────────────────────────────────────────

    def extract_jd(self, jd_text: str) -> ExtractedJD:
        """Parse a job description into structured requirements."""
        logger.info("ScraperAgent: Extracting JD structure...")

        prompt = f"""Extract all information from this job description.

JOB DESCRIPTION:
{jd_text[:8000]}

Return a JSON object with this exact structure:
{{
  "job_title": "exact title from JD",
  "company": "company name",
  "seniority": "junior|mid|senior|lead|principal|executive|unknown",
  "requirements": [
    {{
      "text": "the requirement text",
      "category": "skill|experience|education|soft_skill",
      "importance": "required|preferred|nice_to_have",
      "keywords": ["keyword1", "keyword2"]
    }}
  ],
  "key_skills": ["Python", "AWS", "Machine Learning", ...],
  "responsibilities": ["responsibility 1", "responsibility 2", ...],
  "industry_keywords": ["SaaS", "FinTech", ...],
  "culture_signals": ["fast-paced", "collaborative", ...]
}}"""

        schema = ('{"job_title": "string", "company": "string", "seniority": "junior|mid|senior|lead|principal|executive|unknown",'
                  ' "requirements": [{"text": "string", "category": "skill|experience|education|soft_skill",'
                  ' "importance": "required|preferred|nice_to_have", "keywords": ["string"]}],'
                  ' "key_skills": ["string"], "responsibilities": ["string"],'
                  ' "industry_keywords": ["string"], "culture_signals": ["string"]}')
        data = self.llm.call_json(prompt, system=self.SYSTEM_PROMPT, schema_hint=schema, max_tokens=4096)

        # Map seniority string to enum
        seniority_raw = data.get("seniority", "unknown").lower()
        try:
            seniority = SeniorityLevel(seniority_raw)
        except ValueError:
            seniority = SeniorityLevel.UNKNOWN

        requirements = [
            JDRequirement(
                text=r.get("text", ""),
                category=r.get("category", "skill"),
                importance=r.get("importance", "preferred"),
                keywords=r.get("keywords", []),
            )
            for r in data.get("requirements", [])
        ]

        return ExtractedJD(
            job_title=data.get("job_title", "Unknown Role"),
            company=data.get("company", ""),
            seniority=seniority,
            requirements=requirements,
            key_skills=data.get("key_skills", []),
            responsibilities=data.get("responsibilities", []),
            industry_keywords=data.get("industry_keywords", []),
            culture_signals=data.get("culture_signals", []),
            raw_text=jd_text,
        )

    # ─── Resume Extraction ────────────────────────────────────

    def extract_resume(self, resume_text: str) -> ExtractedResume:
        """Parse a resume into a structured skill inventory."""
        logger.info("ScraperAgent: Extracting resume structure...")

        prompt = f"""Extract all information from this resume. Preserve bullet points verbatim.

RESUME:
{resume_text[:8000]}

Return a JSON object with this exact structure:
{{
  "name": "candidate full name",
  "email": "email@example.com",
  "phone": "phone number",
  "current_title": "most recent job title",
  "summary": "professional summary text",
  "skills": ["Python", "SQL", "Leadership", ...],
  "skill_details": [
    {{
      "skill": "Python",
      "evidence": "the exact bullet point or sentence proving this skill",
      "years": 3.0
    }}
  ],
  "experience": [
    {{
      "title": "Software Engineer",
      "company": "Acme Corp",
      "dates": "Jan 2020 - Present",
      "bullets": ["• Built X using Y...", "• Led team of 5..."]
    }}
  ],
  "education": [
    {{
      "degree": "B.S. Computer Science",
      "school": "MIT",
      "year": "2018"
    }}
  ],
  "certifications": ["AWS Solutions Architect", ...]
}}"""

        data = self.llm.call_json(prompt, system=self.SYSTEM_PROMPT, max_tokens=4096)

        # Build skill_details — only pass dicts; years coercion handled by ResumeSkill validator
        raw_skill_details = data.get("skill_details", [])
        if not isinstance(raw_skill_details, list):
            raw_skill_details = []
        skill_details = [
            ResumeSkill(
                skill=s.get("skill", ""),
                evidence=s.get("evidence", ""),
                years=s.get("years"),
            )
            for s in raw_skill_details
            if isinstance(s, dict)
        ]

        # Normalise experience — ensure bullets is always a list of strings
        raw_experience = data.get("experience", [])
        if not isinstance(raw_experience, list):
            raw_experience = []
        experience = []
        for exp in raw_experience:
            if not isinstance(exp, dict):
                continue
            bullets = exp.get("bullets", [])
            if not isinstance(bullets, list):
                bullets = [str(bullets)] if bullets else []
            else:
                bullets = [str(b) for b in bullets if b is not None]
            experience.append({
                "title": str(exp.get("title", "")),
                "company": str(exp.get("company", "")),
                "dates": str(exp.get("dates", "")),
                "bullets": bullets,
            })

        # skills and certifications — validators in ExtractedResume handle coercion
        return ExtractedResume(
            name=data.get("name", ""),
            email=data.get("email", ""),
            phone=data.get("phone", ""),
            current_title=data.get("current_title", ""),
            summary=data.get("summary", ""),
            skills=data.get("skills", []),
            skill_details=skill_details,
            experience=experience,
            education=data.get("education", []),
            certifications=data.get("certifications", []),
            raw_text=resume_text,
        )
