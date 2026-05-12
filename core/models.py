"""
core/models.py
──────────────
All Pydantic data models shared across agents.

Validators on every List[str] and Optional[float] field guard against the
NVIDIA NIM 8B model occasionally returning dicts, numbers-as-strings, or
other unexpected types inside list/scalar fields.
"""
from __future__ import annotations

import re
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field, field_validator

# ─── Shared normalisation helpers ─────────────────────────────────────────────

def _to_str(v: Any) -> str:
    """Coerce a single value to str, unpacking common dict shapes first."""
    if isinstance(v, dict):
        return str(
            v.get("name") or v.get("skill") or v.get("title") or
            v.get("text") or v.get("value") or v.get("keyword") or
            next(iter(v.values()), "") or v
        )
    return str(v) if v is not None else ""


def _norm_str_list(v: Any) -> List[str]:
    """Normalize any LLM list to List[str], tolerating dicts and None entries."""
    if not isinstance(v, list):
        return []
    return [_to_str(item) for item in v if item is not None]


def _norm_optional_float(v: Any) -> Optional[float]:
    """Normalize to float or None; handles '3 years', 3, 3.5, None."""
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    if isinstance(v, str):
        m = re.search(r"[\d.]+", v)
        return float(m.group()) if m else None
    return None


def _norm_str(v: Any) -> str:
    """Ensure a field that should be str is str (not None, not dict)."""
    if v is None:
        return ""
    if isinstance(v, dict):
        return _to_str(v)
    return str(v)


# ─── Enums ────────────────────────────────────────────────────────────────────

class JobStatus(str, Enum):
    PENDING    = "pending"
    SCRAPING   = "scraping"
    ANALYZING  = "analyzing"
    REWRITING  = "rewriting"
    ASSEMBLING = "assembling"
    COMPLETE   = "complete"
    FAILED     = "failed"


class SeniorityLevel(str, Enum):
    JUNIOR     = "junior"
    MID        = "mid"
    SENIOR     = "senior"
    LEAD       = "lead"
    PRINCIPAL  = "principal"
    EXECUTIVE  = "executive"
    UNKNOWN    = "unknown"


class MatchStrength(str, Enum):
    STRONG  = "strong"
    PARTIAL = "partial"
    MISSING = "missing"


# ─── Scraped Data ─────────────────────────────────────────────────────────────

class JDRequirement(BaseModel):
    """A single extracted requirement from a job description."""
    text: str
    category: str           # "skill", "experience", "education", "soft_skill"
    importance: str         # "required", "preferred", "nice_to_have"
    keywords: List[str] = []

    @field_validator("text", "category", "importance", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("keywords", mode="before")
    @classmethod
    def norm_keywords(cls, v):
        return _norm_str_list(v)


class ExtractedJD(BaseModel):
    """Fully parsed job description."""
    job_title: str
    company: str = ""
    seniority: SeniorityLevel = SeniorityLevel.UNKNOWN
    requirements: List[JDRequirement] = []
    key_skills: List[str] = []
    responsibilities: List[str] = []
    industry_keywords: List[str] = []
    culture_signals: List[str] = []
    raw_text: str = ""

    @field_validator("job_title", "company", "raw_text", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("key_skills", "responsibilities", "industry_keywords",
                     "culture_signals", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)


class ResumeSkill(BaseModel):
    skill: str
    evidence: str
    years: Optional[float] = None

    @field_validator("skill", "evidence", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("years", mode="before")
    @classmethod
    def norm_years(cls, v):
        return _norm_optional_float(v)


class ExtractedResume(BaseModel):
    """Parsed resume content."""
    name: str = ""
    email: str = ""
    phone: str = ""
    current_title: str = ""
    summary: str = ""
    skills: List[str] = []
    skill_details: List[ResumeSkill] = []
    experience: List[Dict[str, Any]] = []
    education: List[Dict[str, Any]] = []
    certifications: List[str] = []
    raw_text: str = ""

    @field_validator("name", "email", "phone", "current_title",
                     "summary", "raw_text", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("skills", "certifications", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)

    @field_validator("experience", "education", mode="before")
    @classmethod
    def norm_dict_lists(cls, v):
        """Ensure these are always List[Dict]; skip any non-dict items."""
        if not isinstance(v, list):
            return []
        return [item if isinstance(item, dict) else {} for item in v]

    @field_validator("skill_details", mode="before")
    @classmethod
    def norm_skill_details(cls, v):
        """Ensure skill_details is a list of dicts (raw dicts get validated by ResumeSkill)."""
        if not isinstance(v, list):
            return []
        return [item for item in v if isinstance(item, dict)]


# ─── Analysis ─────────────────────────────────────────────────────────────────

class SkillGap(BaseModel):
    """A gap between JD requirement and resume content."""
    requirement: str
    category: str
    importance: str
    match: MatchStrength
    resume_evidence: Optional[str] = None
    rewrite_suggestion: Optional[str] = None

    @field_validator("requirement", "category", "importance", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("resume_evidence", "rewrite_suggestion", mode="before")
    @classmethod
    def norm_optional_str(cls, v):
        return _norm_str(v) if v is not None else None


class ATSScore(BaseModel):
    """ATS compatibility score breakdown."""
    overall: float
    keyword_match: float
    formatting: float
    section_completeness: float
    quantification: float
    action_verb_usage: float
    missing_keywords: List[str] = []
    present_keywords: List[str] = []

    @field_validator("missing_keywords", "present_keywords", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)


class AnalysisReport(BaseModel):
    """Full output of the Analyzer Agent."""
    job_id: str
    match_score: float
    ats_score: ATSScore
    seniority_match: bool
    skill_gaps: List[SkillGap] = []
    strengths: List[str] = []
    priority_rewrites: List[str] = []
    strategy_notes: str = ""

    @field_validator("job_id", mode="before")
    @classmethod
    def norm_job_id(cls, v):
        return _norm_str(v)

    @field_validator("strengths", "priority_rewrites", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)

    @field_validator("strategy_notes", mode="before")
    @classmethod
    def norm_strategy_notes(cls, v):
        return _norm_str(v)


# ─── Rewritten Output ─────────────────────────────────────────────────────────

class RewrittenBullet(BaseModel):
    original: str
    rewritten: str
    reason: str

    @field_validator("original", "rewritten", "reason", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)


class RewrittenSection(BaseModel):
    section: str
    original: str
    rewritten: str
    changes_made: List[str] = []

    @field_validator("section", "original", "rewritten", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("changes_made", mode="before")
    @classmethod
    def norm_changes(cls, v):
        return _norm_str_list(v)


class RewrittenResume(BaseModel):
    """Output of the Rewriter Agent."""
    job_id: str
    full_text: str
    sections: List[RewrittenSection] = []
    added_keywords: List[str] = []
    removed_content: List[str] = []
    new_ats_score: Optional[ATSScore] = None

    @field_validator("job_id", "full_text", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("added_keywords", "removed_content", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)


# ─── Application Package ──────────────────────────────────────────────────────

class CoverLetter(BaseModel):
    body: str
    subject_line: str = ""
    tone: str = "professional"

    @field_validator("body", "subject_line", "tone", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)


class ApplicationPackage(BaseModel):
    """Final assembled output from the Application Agent."""
    job_id: str
    cover_letter: CoverLetter
    resume_text: str
    resume_docx_path: Optional[str] = None
    cover_letter_docx_path: Optional[str] = None
    ats_score_before: Optional[float] = None
    ats_score_after: Optional[float] = None
    keywords_added: List[str] = []
    ready_to_submit: bool = False
    submission_notes: str = ""

    @field_validator("job_id", "resume_text", "submission_notes", mode="before")
    @classmethod
    def norm_str_fields(cls, v):
        return _norm_str(v)

    @field_validator("keywords_added", mode="before")
    @classmethod
    def norm_str_lists(cls, v):
        return _norm_str_list(v)


# ─── Self-Critique ────────────────────────────────────────────────────────────

class CritiqueIssue(BaseModel):
    section: str                  # "summary", "experience", "skills"
    issue: str                    # Description of the problem
    severity: str                 # "critical", "major", "minor"
    suggestion: str               # Exact fix to apply

    @field_validator("section", "issue", "severity", "suggestion", mode="before")
    @classmethod
    def norm(cls, v): return _norm_str(v)


class CritiqueReport(BaseModel):
    """Output of the CritiqueAgent — structured feedback on a rewritten resume."""
    passed: bool                       # True = good enough to deliver
    overall_score: float               # 0–100
    keyword_coverage: float            # % of required JD keywords present
    star_compliance: float             # % of bullets using STAR format
    issues: List[CritiqueIssue] = []   # Specific problems found
    praise: List[str] = []             # What is already good
    fix_instructions: str = ""         # Concise rewrite instructions for next pass

    @field_validator("praise", mode="before")
    @classmethod
    def norm_list(cls, v): return _norm_str_list(v)

    @field_validator("fix_instructions", mode="before")
    @classmethod
    def norm_str(cls, v): return _norm_str(v)


# ─── Feedback & Style ─────────────────────────────────────────────────────────

class ApplicationFeedback(BaseModel):
    """User-submitted outcome after applying with this resume."""
    session_id: str
    job_title: str
    company: str
    outcome: str          # "got_interview", "rejected", "no_response", "got_offer"
    notes: str = ""       # What the user thinks went well/poorly
    submitted_at: datetime = Field(default_factory=datetime.utcnow)

    @field_validator("session_id", "job_title", "company", "outcome", "notes", mode="before")
    @classmethod
    def norm(cls, v): return _norm_str(v)


class UserStyleProfile(BaseModel):
    """Learned writing style from the user's own best bullets."""
    sample_bullets: List[str] = []     # User's best bullet points
    tone_keywords: List[str] = []      # Words that characterise their voice
    avg_bullet_length: int = 20        # Words per bullet
    uses_metrics: bool = True
    preferred_verbs: List[str] = []

    @field_validator("sample_bullets", "tone_keywords", "preferred_verbs", mode="before")
    @classmethod
    def norm_lists(cls, v): return _norm_str_list(v)


# ─── Job Session ──────────────────────────────────────────────────────────────

class JobSession(BaseModel):
    """Tracks the full state of a job application session."""
    id: str
    status: JobStatus = JobStatus.PENDING
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    # File paths
    resume_path: Optional[str] = None
    jd_path: Optional[str] = None
    jd_url: Optional[str] = None

    # Agent outputs (populated as pipeline progresses)
    extracted_jd: Optional[ExtractedJD] = None
    extracted_resume: Optional[ExtractedResume] = None
    analysis: Optional[AnalysisReport] = None
    critique: Optional[CritiqueReport] = None
    rewritten_resume: Optional[RewrittenResume] = None
    package: Optional[ApplicationPackage] = None
    style_profile: Optional[UserStyleProfile] = None

    # Error tracking
    error: Optional[str] = None
    agent_logs: List[str] = []

    def log(self, msg: str):
        self.agent_logs.append(f"[{datetime.utcnow().isoformat()}] {msg}")
        self.updated_at = datetime.utcnow()
