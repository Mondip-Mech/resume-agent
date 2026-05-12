"""
eval/evaluator.py
──────────────────
PipelineEvaluator — runs the full agent pipeline against fixture files
and collects objective metrics.

Usage:
    from eval.evaluator import PipelineEvaluator
    ev = PipelineEvaluator()
    result = ev.run_single(jd_text, resume_text)
    print(result)

Or run multiple fixtures:
    results = ev.run_corpus(Path("eval/fixtures"))
"""
from __future__ import annotations

import json
import logging
import sys
import tempfile
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.path.insert(0, str(Path(__file__).parent.parent))

from eval.metrics import (
    aggregate_metrics,
    compute_metrics,
    print_aggregate_report,
    print_report,
)

logger = logging.getLogger(__name__)


class PipelineEvaluator:
    """
    Runs the full job-agent pipeline on (resume, JD) pairs and collects metrics.

    Answers the question: "Does this pipeline objectively improve resumes?"
    """

    def __init__(
        self,
        model: str = "meta/llama-3.1-8b-instruct",
        target_ats: float = 75.0,
        iterations: int = 1,
        output_dir: Optional[str] = None,
    ):
        self.model       = model
        self.target_ats  = target_ats
        self.iterations  = iterations
        self.output_dir  = output_dir or tempfile.mkdtemp(prefix="eval_output_")

    # ─── Single run ───────────────────────────────────────────

    def run_single(
        self,
        jd_text: str,
        resume_text: str,
        label: str = "eval",
    ) -> Dict[str, Any]:
        """
        Run the full pipeline on one (resume, JD) pair.

        Returns
        -------
        Dict of metrics from eval.metrics.compute_metrics()
        """
        import os
        os.environ["LLM_MODEL"]           = self.model
        os.environ["TARGET_ATS_SCORE"]    = str(self.target_ats)
        os.environ["REWRITE_ITERATIONS"]  = str(self.iterations)

        from core.config import get_settings
        get_settings.cache_clear()

        with tempfile.TemporaryDirectory(prefix=f"eval_{label}_") as tmp:
            tmp_path   = Path(tmp)
            upload_dir = tmp_path / "uploads"
            out_dir    = tmp_path / "output"
            upload_dir.mkdir()
            out_dir.mkdir()
            os.environ["UPLOAD_DIR"] = str(upload_dir)
            os.environ["OUTPUT_DIR"] = str(out_dir)
            get_settings.cache_clear()

            # Write files
            resume_path = upload_dir / "resume.txt"
            jd_path     = upload_dir / "jd.txt"
            resume_path.write_text(resume_text, encoding="utf-8")
            jd_path.write_text(jd_text, encoding="utf-8")

            from agents.orchestrator import Orchestrator
            from core.models import JobSession

            session = JobSession(id=str(uuid.uuid4()))
            session.resume_path = str(resume_path)
            session.jd_path     = str(jd_path)

            orch   = Orchestrator()
            orch.app_agent.output_dir = out_dir

            start  = time.perf_counter()
            session = orch.run(session)
            elapsed = time.perf_counter() - start

            metrics = compute_metrics(
                session,
                duration_sec=elapsed,
                token_usage=orch.llm.token_usage,
            )
            metrics["label"] = label
            metrics["model"] = self.model
            metrics["timestamp"] = datetime.utcnow().isoformat()

            return metrics

    # ─── Corpus run ───────────────────────────────────────────

    def run_corpus(
        self,
        fixtures_dir: Path,
        save_results: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Run the pipeline on all (resume_*.txt, jd_*.txt) pairs in fixtures_dir.

        Pair matching: resume_01.txt + jd_01.txt, resume_02.txt + jd_02.txt, etc.
        Also accepts: sample_resume.txt + sample_jd.txt as a single pair.

        Parameters
        ----------
        fixtures_dir : Directory containing fixture files.
        save_results : If True, saves results to eval_results.json in fixtures_dir.

        Returns
        -------
        List of metric dicts, one per fixture pair.
        """
        fixtures_dir = Path(fixtures_dir)
        results      = []

        # Find matching pairs
        pairs = self._find_fixture_pairs(fixtures_dir)
        if not pairs:
            logger.warning(f"No fixture pairs found in {fixtures_dir}")
            return results

        logger.info(f"Running evaluation on {len(pairs)} fixture pair(s)...")

        for i, (resume_path, jd_path, label) in enumerate(pairs, 1):
            logger.info(f"  [{i}/{len(pairs)}] {label}")
            try:
                resume_text = resume_path.read_text(encoding="utf-8")
                jd_text     = jd_path.read_text(encoding="utf-8")
                metrics     = self.run_single(jd_text, resume_text, label=label)
                results.append(metrics)
                print_report(metrics, title=f"Run {i}/{len(pairs)}: {label}")
            except Exception as e:
                logger.error(f"  ✗ {label} failed: {e}")
                results.append({
                    "label": label,
                    "pipeline_status": "error",
                    "error": str(e),
                })

        # Print aggregate
        completed = [r for r in results if r.get("pipeline_status") == "complete"]
        if completed:
            agg = aggregate_metrics(completed)
            print_aggregate_report(agg, title=f"Aggregate ({len(completed)}/{len(pairs)} runs completed)")

        if save_results:
            out_path = fixtures_dir / "eval_results.json"
            out_path.write_text(
                json.dumps({"results": results, "aggregate": aggregate_metrics(completed)}, indent=2),
                encoding="utf-8",
            )
            logger.info(f"Results saved to {out_path}")

        return results

    # ─── Helpers ──────────────────────────────────────────────

    @staticmethod
    def _find_fixture_pairs(fixtures_dir: Path):
        """Find (resume_path, jd_path, label) tuples in fixtures_dir."""
        pairs = []

        # Pattern 1: sample_resume.txt + sample_jd.txt
        sample_resume = fixtures_dir / "sample_resume.txt"
        sample_jd     = fixtures_dir / "sample_jd.txt"
        if sample_resume.exists() and sample_jd.exists():
            pairs.append((sample_resume, sample_jd, "sample"))

        # Pattern 2: resume_01.txt + jd_01.txt, resume_02.txt + jd_02.txt ...
        for resume_file in sorted(fixtures_dir.glob("resume_*.txt")):
            suffix = resume_file.stem.replace("resume_", "")
            jd_file = fixtures_dir / f"jd_{suffix}.txt"
            if jd_file.exists():
                pairs.append((resume_file, jd_file, suffix))

        return pairs

    def compare_models(
        self,
        jd_text: str,
        resume_text: str,
        models: List[str],
    ) -> Dict[str, Dict[str, Any]]:
        """
        Run the same (resume, JD) pair through multiple models and compare results.
        Useful for choosing which model gives the best ATS improvement.

        Returns
        -------
        Dict mapping model_name → metrics dict.
        """
        comparison = {}
        for model in models:
            original_model = self.model
            self.model = model
            logger.info(f"Testing model: {model}")
            try:
                metrics = self.run_single(jd_text, resume_text, label=model)
                comparison[model] = metrics
                print(f"\n  {model}:")
                print(f"    ATS: {metrics.get('ats_score_before')} → {metrics.get('ats_score_after')} "
                      f"(+{metrics.get('ats_improvement_pts')})")
                print(f"    Critique: {metrics.get('critique_score')}/100  "
                      f"passed={metrics.get('critique_passed')}")
            except Exception as e:
                comparison[model] = {"error": str(e)}
                logger.error(f"  {model} failed: {e}")
            finally:
                self.model = original_model

        return comparison
