"""
eval/metrics.py
────────────────
Definitions and computation of all evaluation metrics.

Metrics are computed from a completed JobSession object and returned
as a flat dict that can be logged to JSON, printed, or aggregated.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from core.models import JobSession


def compute_metrics(session: JobSession, duration_sec: float = 0.0,
                    token_usage: Optional[Dict] = None) -> Dict[str, Any]:
    """
    Compute all evaluation metrics from a completed pipeline session.

    Parameters
    ----------
    session       : Completed JobSession (any status, including FAILED).
    duration_sec  : Wall-clock time the pipeline took in seconds.
    token_usage   : Dict from LLMClient.token_usage (input/output/total).

    Returns
    -------
    Flat dict with all metrics. Missing values are None (never raises).
    """
    m: Dict[str, Any] = {
        "session_id":          session.id,
        "pipeline_status":     session.status.value,
        "pipeline_duration_sec": round(duration_sec, 2),
    }

    # ── Token cost ────────────────────────────────────────────
    if token_usage:
        m["llm_input_tokens"]  = token_usage.get("input", 0)
        m["llm_output_tokens"] = token_usage.get("output", 0)
        m["llm_total_tokens"]  = token_usage.get("total", 0)
    else:
        m["llm_input_tokens"]  = None
        m["llm_output_tokens"] = None
        m["llm_total_tokens"]  = None

    # ── Phase 1: Extraction ───────────────────────────────────
    if session.extracted_jd:
        m["jd_requirements_count"] = len(session.extracted_jd.requirements)
        m["jd_skills_count"]       = len(session.extracted_jd.key_skills)
    else:
        m["jd_requirements_count"] = None
        m["jd_skills_count"]       = None

    if session.extracted_resume:
        m["resume_skills_count"]    = len(session.extracted_resume.skills)
        m["resume_experience_roles"] = len(session.extracted_resume.experience)
        m["resume_length_before"]   = len(session.extracted_resume.raw_text)
    else:
        m["resume_skills_count"]    = None
        m["resume_experience_roles"] = None
        m["resume_length_before"]   = None

    # ── Phase 2: Analysis ─────────────────────────────────────
    if session.analysis:
        ats = session.analysis.ats_score
        m["match_score"]             = round(session.analysis.match_score, 1)
        m["ats_score_before"]        = round(ats.overall, 1)
        m["keyword_match_before"]    = round(ats.keyword_match, 1)
        m["quantification_before"]   = round(ats.quantification, 1)
        m["action_verb_before"]      = round(ats.action_verb_usage, 1)
        m["missing_keywords_count"]  = len(ats.missing_keywords)
        m["present_keywords_count"]  = len(ats.present_keywords)
        m["seniority_match"]         = session.analysis.seniority_match
        m["required_gaps_count"]     = sum(
            1 for g in session.analysis.skill_gaps
            if g.importance == "required" and g.match.value == "missing"
        )
        m["partial_gaps_count"]      = sum(
            1 for g in session.analysis.skill_gaps
            if g.match.value == "partial"
        )
    else:
        for k in ["match_score", "ats_score_before", "keyword_match_before",
                  "quantification_before", "action_verb_before",
                  "missing_keywords_count", "present_keywords_count",
                  "seniority_match", "required_gaps_count", "partial_gaps_count"]:
            m[k] = None

    # ── Phase 3: Rewrite ──────────────────────────────────────
    if session.rewritten_resume:
        rwr = session.rewritten_resume
        m["resume_length_after"]    = len(rwr.full_text)
        m["keywords_added_count"]   = len(rwr.added_keywords)
        m["keywords_added_list"]    = rwr.added_keywords[:10]
        m["name_preserved"]         = (
            session.extracted_resume.name.lower() in rwr.full_text.lower()
            if session.extracted_resume and session.extracted_resume.name else None
        )
        if rwr.new_ats_score:
            new = rwr.new_ats_score
            m["ats_score_after"]       = round(new.overall, 1)
            m["keyword_match_after"]   = round(new.keyword_match, 1)
            m["quantification_after"]  = round(new.quantification, 1)
            m["action_verb_after"]     = round(new.action_verb_usage, 1)
            m["ats_improvement_pts"]   = round(
                new.overall - (session.analysis.ats_score.overall if session.analysis else 0), 1
            )
            m["ats_target_reached"]    = new.overall >= 75.0
        else:
            for k in ["ats_score_after", "keyword_match_after", "quantification_after",
                      "action_verb_after", "ats_improvement_pts", "ats_target_reached"]:
                m[k] = None
    else:
        for k in ["resume_length_after", "keywords_added_count", "keywords_added_list",
                  "name_preserved", "ats_score_after", "keyword_match_after",
                  "quantification_after", "action_verb_after",
                  "ats_improvement_pts", "ats_target_reached"]:
            m[k] = None

    # ── Critique ──────────────────────────────────────────────
    if session.critique:
        cr = session.critique
        m["critique_passed"]        = cr.passed
        m["critique_score"]         = round(cr.overall_score, 1)
        m["critique_kw_coverage"]   = round(cr.keyword_coverage, 1)
        m["critique_star_pct"]      = round(cr.star_compliance, 1)
        m["critique_issues_count"]  = len(cr.issues)
        m["critique_critical_count"] = sum(1 for i in cr.issues if i.severity == "critical")
    else:
        for k in ["critique_passed", "critique_score", "critique_kw_coverage",
                  "critique_star_pct", "critique_issues_count", "critique_critical_count"]:
            m[k] = None

    # ── Phase 4: Package ──────────────────────────────────────
    if session.package:
        m["ready_to_submit"]        = session.package.ready_to_submit
        m["cover_letter_generated"] = bool(
            session.package.cover_letter and
            session.package.cover_letter.body and
            "could not be generated" not in session.package.cover_letter.body
        )
        m["resume_docx_generated"]  = bool(
            session.package.resume_docx_path and
            Path(session.package.resume_docx_path).exists()
        )
    else:
        m["ready_to_submit"]        = None
        m["cover_letter_generated"] = None
        m["resume_docx_generated"]  = None

    return m


def aggregate_metrics(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute aggregate statistics across multiple eval runs.

    Parameters
    ----------
    results : List of metric dicts from compute_metrics().

    Returns
    -------
    Dict with mean/std/min/max for numeric fields, and pass-rates for booleans.
    """
    if not results:
        return {}

    numeric_fields = [
        "ats_score_before", "ats_score_after", "ats_improvement_pts",
        "match_score", "missing_keywords_count", "required_gaps_count",
        "critique_score", "critique_kw_coverage", "critique_star_pct",
        "pipeline_duration_sec", "llm_total_tokens",
        "resume_length_before", "resume_length_after",
    ]
    bool_fields = [
        "ats_target_reached", "critique_passed", "ready_to_submit",
        "cover_letter_generated", "seniority_match", "name_preserved",
    ]

    agg: Dict[str, Any] = {"n_runs": len(results)}

    for field in numeric_fields:
        vals = [r[field] for r in results if r.get(field) is not None]
        if vals:
            agg[f"{field}_mean"] = round(sum(vals) / len(vals), 2)
            agg[f"{field}_min"]  = round(min(vals), 2)
            agg[f"{field}_max"]  = round(max(vals), 2)
            if len(vals) > 1:
                mean = agg[f"{field}_mean"]
                variance = sum((v - mean) ** 2 for v in vals) / len(vals)
                agg[f"{field}_std"] = round(variance ** 0.5, 2)

    for field in bool_fields:
        vals = [r[field] for r in results if r.get(field) is not None]
        if vals:
            agg[f"{field}_rate"] = round(sum(vals) / len(vals), 3)

    return agg


def print_report(metrics: Dict[str, Any], title: str = "Pipeline Evaluation") -> None:
    """Pretty-print a single run's metrics to stdout."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}")
    print(f"{'═' * width}")

    sections = [
        ("Pipeline", ["pipeline_status", "pipeline_duration_sec",
                      "llm_total_tokens", "llm_input_tokens", "llm_output_tokens"]),
        ("Extraction", ["jd_requirements_count", "jd_skills_count",
                        "resume_skills_count", "resume_experience_roles"]),
        ("Analysis", ["match_score", "ats_score_before", "keyword_match_before",
                      "missing_keywords_count", "required_gaps_count",
                      "seniority_match"]),
        ("Rewrite", ["ats_score_after", "ats_improvement_pts", "ats_target_reached",
                     "keywords_added_count", "keywords_added_list", "name_preserved"]),
        ("Critique", ["critique_passed", "critique_score", "critique_kw_coverage",
                      "critique_star_pct", "critique_issues_count", "critique_critical_count"]),
        ("Package", ["ready_to_submit", "cover_letter_generated", "resume_docx_generated"]),
    ]

    for section_name, fields in sections:
        print(f"\n  ── {section_name} {'─' * (width - len(section_name) - 6)}")
        for field in fields:
            val = metrics.get(field)
            if val is None:
                continue
            label = field.replace("_", " ").title()
            if isinstance(val, bool):
                icon = "✅" if val else "❌"
                print(f"    {label:<35} {icon}")
            elif isinstance(val, list):
                print(f"    {label:<35} {', '.join(str(v) for v in val[:6])}")
            elif isinstance(val, float):
                print(f"    {label:<35} {val:.1f}")
            else:
                print(f"    {label:<35} {val}")

    print(f"\n{'═' * width}\n")


def print_aggregate_report(agg: Dict[str, Any], title: str = "Aggregate Evaluation") -> None:
    """Pretty-print aggregate statistics across multiple runs."""
    width = 60
    print(f"\n{'=' * width}")
    print(f"  {title}  (n={agg.get('n_runs', '?')} runs)")
    print(f"{'═' * width}")

    key_metrics = [
        ("ATS Before (mean)", "ats_score_before_mean"),
        ("ATS After (mean)", "ats_score_after_mean"),
        ("ATS Improvement (mean)", "ats_improvement_pts_mean"),
        ("Target ATS Reached", "ats_target_reached_rate"),
        ("Critique Pass Rate", "critique_passed_rate"),
        ("Critique Score (mean)", "critique_score_mean"),
        ("Ready to Submit Rate", "ready_to_submit_rate"),
        ("Avg Duration (sec)", "pipeline_duration_sec_mean"),
        ("Avg Tokens Used", "llm_total_tokens_mean"),
    ]
    for label, key in key_metrics:
        val = agg.get(key)
        if val is None:
            continue
        if "rate" in key:
            print(f"  {label:<35} {val * 100:.1f}%")
        else:
            print(f"  {label:<35} {val:.1f}")

    print(f"\n{'═' * width}\n")
