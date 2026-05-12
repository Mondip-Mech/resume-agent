"""
eval/run_eval.py
─────────────────
CLI entry point for running the evaluation framework.

Usage:
    # Run on sample fixtures (uses NVIDIA API — costs tokens):
    python eval/run_eval.py

    # Run on a specific resume + JD:
    python eval/run_eval.py --resume path/to/resume.txt --jd path/to/jd.txt

    # Compare multiple models:
    python eval/run_eval.py --compare-models

    # Dry run (no API calls — just verify fixtures load correctly):
    python eval/run_eval.py --dry-run
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("eval")

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def cmd_run_fixtures(args):
    """Run evaluation on all fixture pairs."""
    from dotenv import load_dotenv
    load_dotenv()
    from eval.evaluator import PipelineEvaluator
    ev = PipelineEvaluator(
        model=args.model,
        target_ats=args.target_ats,
        iterations=args.iterations,
    )
    results = ev.run_corpus(FIXTURES_DIR, save_results=True)
    if args.json:
        print(json.dumps(results, indent=2))


def cmd_run_single(args):
    """Run on a specific resume + JD file."""
    from dotenv import load_dotenv
    load_dotenv()
    resume_text = Path(args.resume).read_text(encoding="utf-8")
    jd_text     = Path(args.jd).read_text(encoding="utf-8")
    from eval.evaluator import PipelineEvaluator
    ev = PipelineEvaluator(model=args.model, target_ats=args.target_ats)
    metrics = ev.run_single(jd_text, resume_text, label="custom")
    from eval.metrics import print_report
    print_report(metrics, title="Custom Evaluation Run")
    if args.json:
        print(json.dumps(metrics, indent=2))


def cmd_compare_models(args):
    """Compare 2+ models on the sample fixture."""
    from dotenv import load_dotenv
    load_dotenv()
    sample_resume = (FIXTURES_DIR / "sample_resume.txt").read_text(encoding="utf-8")
    sample_jd     = (FIXTURES_DIR / "sample_jd.txt").read_text(encoding="utf-8")
    models = [
        "meta/llama-3.1-8b-instruct",
        "meta/llama-3.1-70b-instruct",
    ]
    from eval.evaluator import PipelineEvaluator
    ev = PipelineEvaluator()
    comparison = ev.compare_models(sample_jd, sample_resume, models=models)
    print("\n-- Model Comparison -------------------------------------")
    for model, m in comparison.items():
        if "error" in m:
            print(f"  {model}: ERROR — {m['error']}")
        else:
            print(f"  {model}:")
            print(f"    ATS: {m.get('ats_score_before')} → {m.get('ats_score_after')} "
                  f"(Δ {m.get('ats_improvement_pts'):+.1f})")
            print(f"    Critique: {m.get('critique_score')}/100 | "
                  f"Tokens: {m.get('llm_total_tokens')} | "
                  f"Time: {m.get('pipeline_duration_sec'):.0f}s")


def cmd_dry_run(_args):
    """Verify fixtures load correctly without making API calls."""
    print("-- Dry Run ----------------------------------------------")
    sample_resume = FIXTURES_DIR / "sample_resume.txt"
    sample_jd     = FIXTURES_DIR / "sample_jd.txt"

    if not sample_resume.exists():
        print(f"  ✗ Missing: {sample_resume}")
        sys.exit(1)
    if not sample_jd.exists():
        print(f"  ✗ Missing: {sample_jd}")
        sys.exit(1)

    resume_text = sample_resume.read_text(encoding="utf-8")
    jd_text     = sample_jd.read_text(encoding="utf-8")
    print(f"  ✓ Resume loaded: {len(resume_text)} chars")
    print(f"  ✓ JD loaded: {len(jd_text)} chars")

    # Test ATS scorer (no API needed)
    from tools.ats_scorer import extract_jd_keywords, score_resume
    keywords = extract_jd_keywords(jd_text, top_k=15)
    score    = score_resume(resume_text, keywords)
    print(f"  ✓ ATS scorer: {score.overall:.1f}/100 "
          f"(keywords: {score.keyword_match:.0f}%, "
          f"quant: {score.quantification:.0f}%)")
    print(f"  ✓ Keywords extracted: {', '.join(keywords[:8])}")
    print("\n  Baseline (no rewrite):")
    print(f"    ATS score:      {score.overall:.1f}/100")
    print(f"    Keyword match:  {score.keyword_match:.0f}%")
    print(f"    Missing:        {', '.join(score.missing_keywords[:5])}")
    print(f"    Present:        {', '.join(score.present_keywords[:5])}")
    print("\n  ✓ Dry run passed — all fixtures valid, ATS scorer working.")
    print("  Run without --dry-run to execute the full pipeline (requires NVIDIA_API_KEY).")


# ─── CLI ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate the job-agent pipeline objectively.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--model", default="meta/llama-3.1-8b-instruct",
                        help="NVIDIA NIM model to use for evaluation")
    parser.add_argument("--target-ats", type=float, default=75.0,
                        help="Target ATS score (default: 75.0)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Max rewrite iterations (default: 1)")
    parser.add_argument("--resume", help="Path to resume .txt file")
    parser.add_argument("--jd", help="Path to job description .txt file")
    parser.add_argument("--compare-models", action="store_true",
                        help="Compare 8B vs 70B models on sample fixture")
    parser.add_argument("--dry-run", action="store_true",
                        help="Validate fixtures and scorer without API calls")
    parser.add_argument("--json", action="store_true",
                        help="Also print results as JSON")

    args = parser.parse_args()

    if args.dry_run:
        cmd_dry_run(args)
    elif args.compare_models:
        cmd_compare_models(args)
    elif args.resume and args.jd:
        cmd_run_single(args)
    else:
        cmd_run_fixtures(args)


if __name__ == "__main__":
    main()
