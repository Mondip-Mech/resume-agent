#!/usr/bin/env python3
"""
scripts/run_agent.py
─────────────────────
Command-line interface for the Job Application Agent.

Usage:
  python scripts/run_agent.py \\
    --resume ./data/my_resume.pdf \\
    --jd ./data/job_description.txt \\
    --output ./output/

  python scripts/run_agent.py \\
    --resume ./data/resume.docx \\
    --jd-url https://jobs.example.com/posting/12345 \\
    --output ./output/
"""
import argparse
import asyncio
import json
import logging
import sys
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
)


async def main(args: argparse.Namespace):
    from agents.orchestrator import Orchestrator
    from core.config import get_settings
    from core.models import JobSession
    from parsers.pdf_parser import extract_from_url

    get_settings()  # validate env vars early — raises if NVIDIA_API_KEY is missing
    Path(args.output).mkdir(parents=True, exist_ok=True)

    session = JobSession(id=str(uuid.uuid4()))

    # ── Set up file paths ──────────────────────────────────────
    if args.resume:
        session.resume_path = args.resume
    else:
        print("ERROR: --resume is required")
        sys.exit(1)

    if args.jd:
        session.jd_path = args.jd
    elif args.jd_url:
        print(f"Scraping JD from URL: {args.jd_url}")
        jd_text = await extract_from_url(args.jd_url)
        jd_path = Path(args.output) / "jd_scraped.txt"
        jd_path.write_text(jd_text)
        session.jd_path = str(jd_path)
        print(f"JD scraped: {len(jd_text)} characters")
    else:
        print("ERROR: Either --jd or --jd-url is required")
        sys.exit(1)

    # ── Run pipeline ──────────────────────────────────────────
    print(f"\n{'='*60}")
    print("  AI Job Application Agent")
    print(f"  Session: {session.id[:8]}...")
    print(f"{'='*60}\n")

    orchestrator = Orchestrator()
    # Override output dir to CLI arg
    orchestrator.app_agent.output_dir = Path(args.output)

    session = orchestrator.run(session)

    # ── Print results ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PIPELINE RESULTS — {session.status.value.upper()}")
    print(f"{'='*60}")

    if session.status.value == "failed":
        print(f"\n❌ FAILED: {session.error}")
        return

    if session.extracted_jd:
        print(f"\n📋 Role: {session.extracted_jd.job_title} at {session.extracted_jd.company}")

    if session.analysis:
        ats = session.analysis.ats_score
        print("\n📊 Analysis:")
        print(f"   Match score:    {session.analysis.match_score:.0f}/100")
        print(f"   ATS (before):   {ats.overall:.0f}/100")
        print(f"   Keyword match:  {ats.keyword_match:.0f}%")
        print(f"   Quantification: {ats.quantification:.0f}%")
        print(f"   Missing:        {', '.join(ats.missing_keywords[:5])}")

    if session.rewritten_resume and session.rewritten_resume.new_ats_score:
        new_ats = session.rewritten_resume.new_ats_score
        old_ats = session.analysis.ats_score.overall
        delta = new_ats.overall - old_ats
        print("\n✍️  Rewrite:")
        print(f"   ATS (after):  {new_ats.overall:.0f}/100  ({'+' if delta >= 0 else ''}{delta:.1f})")
        print(f"   Keywords added: {', '.join(session.rewritten_resume.added_keywords[:6])}")

    if session.package:
        pkg = session.package
        print("\n📦 Package:")
        print(f"   Ready: {'✅' if pkg.ready_to_submit else '⚠️  ' + pkg.submission_notes[:80]}")
        if pkg.resume_docx_path:
            print(f"   Resume:       {pkg.resume_docx_path}")
        if pkg.cover_letter_docx_path:
            print(f"   Cover letter: {pkg.cover_letter_docx_path}")

    # Print agent log
    if args.verbose:
        print("\n📜 Agent Log:")
        for entry in session.agent_logs:
            print(f"   {entry}")

    # Save JSON report
    report_path = Path(args.output) / f"{session.id[:8]}_report.json"
    report = {
        "session_id": session.id,
        "status": session.status.value,
        "job_title": session.extracted_jd.job_title if session.extracted_jd else None,
        "company": session.extracted_jd.company if session.extracted_jd else None,
        "ats_before": session.analysis.ats_score.overall if session.analysis else None,
        "ats_after": session.rewritten_resume.new_ats_score.overall if session.rewritten_resume and session.rewritten_resume.new_ats_score else None,
        "match_score": session.analysis.match_score if session.analysis else None,
        "keywords_added": session.rewritten_resume.added_keywords if session.rewritten_resume else [],
        "priority_rewrites": session.analysis.priority_rewrites if session.analysis else [],
        "ready_to_submit": session.package.ready_to_submit if session.package else False,
        "token_usage": orchestrator.llm.token_usage,
    }
    report_path.write_text(json.dumps(report, indent=2))
    print(f"\n💾 JSON report saved: {report_path}")
    print(f"{'='*60}\n")


def parse_args():
    parser = argparse.ArgumentParser(
        description="AI Job Application Agent — Resume Optimizer",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--resume", required=True, help="Path to resume (PDF or DOCX)")
    jd_group = parser.add_mutually_exclusive_group(required=True)
    jd_group.add_argument("--jd", help="Path to job description (PDF or TXT)")
    jd_group.add_argument("--jd-url", help="URL of the job posting to scrape")
    parser.add_argument("--output", default="./output", help="Output directory")
    parser.add_argument("--verbose", action="store_true", help="Print full agent log")
    return parser.parse_args()


if __name__ == "__main__":
    asyncio.run(main(parse_args()))
