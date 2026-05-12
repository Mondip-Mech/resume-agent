"""
tests/test_memory.py
─────────────────────
Tests for core/memory.py — TF-IDF store, RAG retrieval,
feedback loop, style learning, market intelligence, persistence.

All tests use a tmp_path-based AgentMemory so nothing writes to disk
outside the test's temp directory.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ─── TF-IDF store (internal) ─────────────────────────────────────────────────

class TestTFIDFStore:

    def _make_store(self):
        from core.memory import _TFIDFStore
        return _TFIDFStore()

    def test_upsert_and_query_basic(self):
        store = self._make_store()
        store.upsert(
            documents=["Python machine learning expert", "Java backend developer", "SQL data analyst"],
            ids=["d1", "d2", "d3"],
        )
        results = store.query(query_texts=["Python ML"], n_results=1)
        top_doc = results["documents"][0][0]
        assert "Python" in top_doc

    def test_upsert_deduplicates_by_id(self):
        store = self._make_store()
        store.upsert(documents=["First version"], ids=["doc1"])
        store.upsert(documents=["Updated version"], ids=["doc1"])
        assert store.count() == 1
        results = store.query(query_texts=["version"], n_results=1)
        assert "Updated" in results["documents"][0][0]

    def test_query_empty_store_returns_empty(self):
        store = self._make_store()
        results = store.query(query_texts=["anything"], n_results=3)
        assert results["documents"] == [[]]

    def test_cosine_identical_vectors_is_one(self):
        from collections import Counter

        from core.memory import _TFIDFStore
        v = Counter({"python": 2, "ml": 1})
        assert _TFIDFStore._cosine(v, v) == pytest.approx(1.0)

    def test_cosine_empty_vectors_is_zero(self):
        from collections import Counter

        from core.memory import _TFIDFStore
        assert _TFIDFStore._cosine(Counter(), Counter({"a": 1})) == 0.0

    def test_query_returns_at_most_n_results(self):
        store = self._make_store()
        store.upsert(
            documents=[f"document {i}" for i in range(10)],
            ids=[f"id{i}" for i in range(10)],
        )
        results = store.query(query_texts=["document"], n_results=3)
        assert len(results["documents"][0]) == 3


# ─── AgentMemory — JD / Resume chunks ────────────────────────────────────────

class TestAgentMemoryChunks:

    def test_store_and_search_jd(self, tmp_memory):
        tmp_memory.store_jd("sess1", "Senior Python Developer at TechCorp. Requires AWS and Docker.")
        results = tmp_memory.search_jd("sess1", "Python AWS", top_k=2)
        assert isinstance(results, list)
        assert len(results) > 0
        assert any("Python" in r or "AWS" in r for r in results)

    def test_store_and_search_resume(self, tmp_memory):
        tmp_memory.store_resume("sess1", "Jane Smith. Python expert. Built ML pipelines at Google.")
        results = tmp_memory.search_resume("sess1", "machine learning", top_k=2)
        assert isinstance(results, list)

    def test_search_nonexistent_session_returns_empty(self, tmp_memory):
        results = tmp_memory.search_jd("nonexistent_session", "Python", top_k=3)
        assert results == []

    def test_different_sessions_are_isolated(self, tmp_memory):
        tmp_memory.store_jd("sess_a", "Java Spring Boot microservices")
        tmp_memory.store_jd("sess_b", "Python TensorFlow machine learning")
        results = tmp_memory.search_jd("sess_a", "Python TensorFlow", top_k=2)
        # sess_a only has Java content, should not return TensorFlow docs
        for r in results:
            assert "TensorFlow" not in r


# ─── AgentMemory — Bullet-level RAG ──────────────────────────────────────────

class TestBulletRAG:

    def test_store_bullets_and_retrieve(self, tmp_memory):
        bullets = [
            "Built XGBoost model achieving 92% accuracy",
            "Led team of 5 engineers on ML platform",
            "Designed ETL pipeline processing 5TB daily",
            "Reduced model inference latency by 40%",
            "Deployed microservices on AWS ECS",
        ]
        tmp_memory.store_bullets("sess1", bullets)
        results = tmp_memory.retrieve_relevant_bullets("sess1", "machine learning model", top_k=2)
        assert isinstance(results, list)
        assert len(results) <= 2

    def test_store_bullets_empty_list_is_safe(self, tmp_memory):
        tmp_memory.store_bullets("sess1", [])  # should not raise

    def test_extract_and_store_bullets_from_resume(self, tmp_memory, sample_resume_text):
        bullets = tmp_memory.extract_and_store_bullets("sess1", sample_resume_text)
        assert isinstance(bullets, list)
        assert len(bullets) > 0
        # All extracted items should be non-empty strings
        assert all(isinstance(b, str) and len(b) > 0 for b in bullets)

    def test_extract_bullets_detects_action_verb_lines(self, tmp_memory):
        resume = (
            "EXPERIENCE\n"
            "Senior Engineer | Acme | 2020-2024\n"
            "Built distributed system serving 10M users daily\n"
            "Optimized database queries reducing latency by 60%\n"
            "Deployed containerized services on Kubernetes\n"
            "Some random sentence that is not a bullet.\n"
        )
        bullets = tmp_memory.extract_and_store_bullets("sess1", resume)
        # At least 2 action-verb lines should be detected
        assert len(bullets) >= 2
        verb_starts = {"built", "optimized", "deployed"}
        found_verbs = {b.split()[0].lower() for b in bullets if b.split()}
        assert len(found_verbs & verb_starts) >= 1

    def test_bullets_added_to_style_store(self, tmp_memory):
        bullets = ["Built ML pipeline", "Reduced latency by 30%"]
        tmp_memory.store_bullets("sess1", bullets)
        style = tmp_memory.get_style_context(top_k=5)
        assert len(style) > 0  # style store was populated


# ─── AgentMemory — Feedback Loop ─────────────────────────────────────────────

class TestFeedbackLoop:

    def test_store_and_retrieve_feedback(self, tmp_memory):
        tmp_memory.store_feedback(
            session_id="sess1",
            job_title="Data Scientist",
            company="Acme",
            outcome="got_interview",
            notes="Recruiter liked the ML experience",
        )
        all_fb = tmp_memory.get_all_feedback()
        assert len(all_fb) == 1
        assert all_fb[0]["outcome"] == "got_interview"
        assert all_fb[0]["company"] == "Acme"

    def test_feedback_insights_mentions_total_count(self, tmp_memory):
        tmp_memory.store_feedback("s1", "ML Engineer", "Google", "got_interview")
        tmp_memory.store_feedback("s2", "ML Engineer", "Meta", "rejected")
        tmp_memory.store_feedback("s3", "Data Scientist", "Apple", "no_response")
        insights = tmp_memory.get_feedback_insights()
        assert "3" in insights  # total count mentioned

    def test_feedback_insights_empty_returns_empty_string(self, tmp_memory):
        assert tmp_memory.get_feedback_insights() == ""

    def test_feedback_insights_filters_by_similar_role(self, tmp_memory):
        tmp_memory.store_feedback("s1", "ML Engineer", "Google", "got_interview",
                                  notes="Liked Python background")
        tmp_memory.store_feedback("s2", "Frontend Dev", "Meta", "rejected",
                                  notes="React skills too weak")
        insights = tmp_memory.get_feedback_insights(similar_role="ML Engineer")
        assert "Python" in insights  # relevant note surfaced
        # Frontend notes should not appear
        assert "React" not in insights

    def test_multiple_feedback_entries(self, tmp_memory):
        for i in range(5):
            tmp_memory.store_feedback(f"s{i}", "Data Scientist", f"Company{i}", "got_interview")
        assert len(tmp_memory.get_all_feedback()) == 5


# ─── AgentMemory — Style Learning ────────────────────────────────────────────

class TestStyleLearning:

    def test_store_and_get_style_context(self, tmp_memory):
        bullets = [
            "Built distributed ML platform serving 50M users",
            "Reduced model training time by 60% via parallelization",
        ]
        tmp_memory.store_style_samples(bullets)
        context = tmp_memory.get_style_context(top_k=5)
        assert "USER'S WRITING STYLE" in context
        assert "Built" in context or "Reduced" in context

    def test_style_context_empty_when_no_samples(self, tmp_memory):
        assert tmp_memory.get_style_context() == ""

    def test_analyze_style_has_expected_keys(self, tmp_memory):
        tmp_memory.store_style_samples([
            "Built API serving 1M requests/day",
            "Reduced latency by 40% using caching",
            "Led team of 5 engineers",
        ])
        stats = tmp_memory.analyze_style()
        assert "avg_bullet_length" in stats
        assert "uses_metrics" in stats
        assert "top_verbs" in stats
        assert "total_samples" in stats
        assert stats["total_samples"] == 3

    def test_style_deduplication(self, tmp_memory):
        bullets = ["Built ML pipeline", "Built ML pipeline"]  # duplicate
        tmp_memory.store_style_samples(bullets)
        # Duplicates should be removed
        assert len(tmp_memory._style_bullets) == 1


# ─── AgentMemory — Market Intelligence ───────────────────────────────────────

class TestMarketIntelligence:

    def test_store_posting_and_get_keywords(self, tmp_memory):
        jd1 = "Python machine learning TensorFlow AWS Docker Kubernetes SQL BigQuery"
        jd2 = "Python data science TensorFlow scikit-learn AWS Spark ETL"
        tmp_memory.store_job_posting(jd1)
        tmp_memory.store_job_posting(jd2)
        keywords = tmp_memory.get_market_keywords(top_k=10)
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        # Python and TensorFlow appear in both JDs — should be high up
        lower_kw = [k.lower() for k in keywords]
        assert "python" in lower_kw or "tensorflow" in lower_kw

    def test_market_keywords_empty_when_no_postings(self, tmp_memory):
        assert tmp_memory.get_market_keywords() == []

    def test_market_summary_mentions_count(self, tmp_memory):
        tmp_memory.store_job_posting("Python SQL AWS machine learning")
        tmp_memory.store_job_posting("Python Docker Kubernetes CI/CD")
        summary = tmp_memory.get_market_summary()
        assert "2" in summary
        assert "Market Intelligence" in summary

    def test_market_summary_empty_when_no_postings(self, tmp_memory):
        assert tmp_memory.get_market_summary() == ""

    def test_stop_words_filtered_from_keywords(self, tmp_memory):
        # Common stop words should never appear in keyword results
        jd = "The candidate will work with the team. They will manage the project and the experience."
        tmp_memory.store_job_posting(jd)
        keywords = tmp_memory.get_market_keywords(top_k=20)
        stop = {"the", "and", "will", "with", "they", "experience"}
        result_lower = {k.lower() for k in keywords}
        assert len(result_lower & stop) == 0


# ─── AgentMemory — Persistence ───────────────────────────────────────────────

class TestPersistence:

    def test_save_and_reload_feedback(self, tmp_path):
        from core.memory import AgentMemory
        mem1 = AgentMemory(persist_dir=str(tmp_path))
        mem1.store_feedback("s1", "Data Scientist", "Acme", "got_interview", "Great fit")
        mem1.store_style_samples(["Built ML pipeline serving 10M users"])

        # Create a second instance pointing at the same dir
        mem2 = AgentMemory(persist_dir=str(tmp_path))
        assert len(mem2.get_all_feedback()) == 1
        assert mem2.get_all_feedback()[0]["outcome"] == "got_interview"
        assert len(mem2._style_bullets) == 1

    def test_persist_file_is_valid_json(self, tmp_path):
        from core.memory import AgentMemory
        mem = AgentMemory(persist_dir=str(tmp_path))
        mem.store_feedback("s1", "Engineer", "Corp", "rejected")
        persist_file = tmp_path / "agent_memory.json"
        assert persist_file.exists()
        data = json.loads(persist_file.read_text(encoding="utf-8"))
        assert "feedback" in data
        assert "style_bullets" in data
        assert "market_jobs" in data

    def test_handles_missing_persist_file_gracefully(self, tmp_path):
        from core.memory import AgentMemory
        # Fresh directory — no file yet. Should not raise.
        mem = AgentMemory(persist_dir=str(tmp_path / "fresh_dir"))
        assert mem.get_all_feedback() == []


# ─── AgentMemory — Session cleanup ───────────────────────────────────────────

class TestSessionCleanup:

    def test_clear_session_does_not_raise(self, tmp_memory):
        tmp_memory.store_jd("sess1", "Python AWS machine learning")
        tmp_memory.store_resume("sess1", "Jane Smith Python developer")
        tmp_memory.clear_session("sess1")  # should not raise

    def test_search_after_clear_returns_empty(self, tmp_memory):
        tmp_memory.store_jd("sess1", "Python AWS machine learning")
        tmp_memory.clear_session("sess1")
        results = tmp_memory.search_jd("sess1", "Python", top_k=3)
        assert results == []
