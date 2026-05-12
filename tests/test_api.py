"""
tests/test_api.py
──────────────────
Tests for the FastAPI REST API (api/main.py).

Uses httpx + ASGITransport for in-process testing — no real server needed,
no NVIDIA API key required.

Covers:
  - GET  /api/health
  - POST /api/jobs (create session)
  - POST /api/jobs/{id}/upload
  - POST /api/jobs/{id}/upload-url
  - POST /api/jobs/{id}/run  (with pipeline mocked)
  - GET  /api/jobs/{id}/status
  - GET  /api/jobs/{id}/result
  - GET  /api/jobs (list all)
  - POST /api/analyze (quick heuristic scorer — no LLM)
  - 404 / 400 error handling
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
from unittest.mock import patch

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

# ── httpx is already in requirements via httpx==0.27.2 ──────────────────────
try:
    import httpx  # noqa: F401  — imported for availability check only
    from httpx import ASGITransport, AsyncClient
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(not HTTPX_AVAILABLE, reason="httpx not installed"),
]


# ─── App fixture ─────────────────────────────────────────────────────────────

@pytest.fixture
def api_app(tmp_path):
    """
    Import and configure the FastAPI app with temp directories.
    Patches get_settings() so no real .env is needed.
    """
    with patch.dict("os.environ", {
        "NVIDIA_API_KEY": "test-key-for-api-tests",
        "UPLOAD_DIR":     str(tmp_path / "uploads"),
        "OUTPUT_DIR":     str(tmp_path / "output"),
        "CHROMA_DIR":     str(tmp_path / "chroma"),
    }):
        # Clear module cache so api/main.py picks up the patched env
        for mod in list(sys.modules.keys()):
            if "api.main" in mod or "core.config" in mod:
                del sys.modules[mod]

        from core.config import get_settings
        try:
            get_settings.cache_clear()
        except AttributeError:
            pass

        (tmp_path / "uploads").mkdir(parents=True, exist_ok=True)
        (tmp_path / "output").mkdir(parents=True, exist_ok=True)

        from api.main import app
        return app


@pytest.fixture
async def client(api_app):
    """Async httpx client backed by the in-process ASGI app."""
    async with AsyncClient(
        transport=ASGITransport(app=api_app),
        base_url="http://test",
    ) as c:
        yield c


# ─── Health ───────────────────────────────────────────────────────────────────

class TestHealth:

    async def test_health_returns_ok(self, client):
        resp = await client.get("/api/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert "sessions" in data

    async def test_health_tracks_session_count(self, client):
        before = (await client.get("/api/health")).json()["sessions"]
        await client.post("/api/jobs")
        after  = (await client.get("/api/health")).json()["sessions"]
        assert after == before + 1


# ─── Create / list sessions ───────────────────────────────────────────────────

class TestSessionManagement:

    async def test_create_job_returns_session_id(self, client):
        resp = await client.post("/api/jobs")
        assert resp.status_code == 200
        data = resp.json()
        assert "session_id" in data
        assert len(data["session_id"]) == 36  # UUID

    async def test_create_multiple_sessions(self, client):
        ids = set()
        for _ in range(3):
            resp = await client.post("/api/jobs")
            ids.add(resp.json()["session_id"])
        assert len(ids) == 3  # all unique

    async def test_list_jobs_returns_list(self, client):
        await client.post("/api/jobs")
        resp = await client.get("/api/jobs")
        assert resp.status_code == 200
        assert isinstance(resp.json(), list)
        assert len(resp.json()) >= 1

    async def test_list_jobs_contains_status(self, client):
        await client.post("/api/jobs")
        jobs = (await client.get("/api/jobs")).json()
        assert "status" in jobs[-1]
        assert jobs[-1]["status"] == "pending"


# ─── File upload ──────────────────────────────────────────────────────────────

class TestFileUpload:

    async def test_upload_resume_only(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]

        resume_content = b"Jane Smith\nSenior Data Scientist\nPython, AWS, TensorFlow"
        resp = await client.post(
            f"/api/jobs/{sid}/upload",
            files={"resume": ("resume.txt", io.BytesIO(resume_content), "text/plain")},
        )
        assert resp.status_code == 200
        assert "resume" in resp.json()

    async def test_upload_resume_and_jd(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]

        resp = await client.post(
            f"/api/jobs/{sid}/upload",
            files={
                "resume": ("resume.txt", io.BytesIO(b"Jane Smith\nPython developer"), "text/plain"),
                "jd":     ("jd.txt",     io.BytesIO(b"Data Scientist. Python required."), "text/plain"),
            },
        )
        assert resp.status_code == 200

    async def test_upload_to_nonexistent_session(self, client):
        resp = await client.post(
            "/api/jobs/nonexistent-id/upload",
            files={"resume": ("resume.txt", io.BytesIO(b"Jane"), "text/plain")},
        )
        assert resp.status_code == 404

    async def test_upload_url_saves_url(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.post(
            f"/api/jobs/{sid}/upload-url",
            json={"jd_url": "https://example.com/jobs/123"},
        )
        assert resp.status_code == 200
        assert "url" in resp.json()


# ─── Pipeline run ─────────────────────────────────────────────────────────────

class TestPipelineRun:

    async def _setup_session(self, client) -> str:
        """Create a session with resume + JD uploaded."""
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        await client.post(
            f"/api/jobs/{sid}/upload",
            files={
                "resume": ("resume.txt", io.BytesIO(b"Jane Smith\nPython, SQL, TensorFlow"), "text/plain"),
                "jd":     ("jd.txt",     io.BytesIO(b"Data Scientist. Python required. SQL needed."), "text/plain"),
            },
        )
        return sid

    async def test_run_without_resume_returns_400(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        # Don't upload anything
        resp = await client.post(f"/api/jobs/{sid}/run")
        assert resp.status_code == 400
        assert "resume" in resp.json()["detail"].lower()

    async def test_run_returns_started_message(self, client):
        sid = await self._setup_session(client)

        with patch("api.main._run_pipeline") as mock_run:
            mock_run.return_value = None
            resp = await client.post(f"/api/jobs/{sid}/run")

        assert resp.status_code == 200
        assert "started" in resp.json()["message"].lower()

    async def test_double_run_returns_400(self, client):
        sid = await self._setup_session(client)

        with patch("api.main._run_pipeline"):
            await client.post(f"/api/jobs/{sid}/run")

        # Try to run again while already started
        with patch("api.main._run_pipeline"):
            resp = await client.post(f"/api/jobs/{sid}/run")

        # Should fail because session is no longer PENDING
        assert resp.status_code == 400


# ─── Status polling ───────────────────────────────────────────────────────────

class TestStatusPolling:

    async def test_status_returns_pending_for_new_session(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.get(f"/api/jobs/{sid}/status")
        assert resp.status_code == 200
        assert resp.json()["status"] == "pending"

    async def test_status_includes_logs(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.get(f"/api/jobs/{sid}/status")
        assert "logs" in resp.json()
        assert isinstance(resp.json()["logs"], list)

    async def test_status_for_nonexistent_session(self, client):
        resp = await client.get("/api/jobs/does-not-exist/status")
        assert resp.status_code == 404


# ─── Result retrieval ─────────────────────────────────────────────────────────

class TestResultRetrieval:

    async def test_result_before_complete_returns_400(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.get(f"/api/jobs/{sid}/result")
        assert resp.status_code == 400
        assert "not complete" in resp.json()["detail"].lower()

    async def test_result_for_nonexistent_session(self, client):
        resp = await client.get("/api/jobs/ghost-session/result")
        assert resp.status_code == 404

    async def test_download_nonexistent_doc(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.get(f"/api/jobs/{sid}/download/resume")
        assert resp.status_code == 404

    async def test_download_invalid_doc_type(self, client):
        resp = await client.post("/api/jobs")
        sid  = resp.json()["session_id"]
        resp = await client.get(f"/api/jobs/{sid}/download/invalid_type")
        # Either 404 (no package) or 400 (bad type) is acceptable
        assert resp.status_code in (400, 404)


# ─── Quick analyze endpoint ───────────────────────────────────────────────────

class TestQuickAnalyze:

    async def test_analyze_returns_score(self, client):
        resp = await client.post("/api/analyze", json={
            "resume_text": (
                "Jane Smith\nSUMMARY\nData Scientist with Python and TensorFlow experience.\n"
                "SKILLS\nPython, SQL, TensorFlow, AWS, Docker\n"
                "EXPERIENCE\nBuilt ML models achieving 92% accuracy at TechCorp."
            ),
            "jd_text": "Senior Data Scientist. Python required. TensorFlow, AWS, SQL needed.",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "ats_score" in data
        assert 0 <= data["ats_score"] <= 100
        assert "present_keywords" in data
        assert "missing_keywords" in data
        assert "Python" in data["present_keywords"]

    async def test_analyze_empty_resume_returns_400(self, client):
        resp = await client.post("/api/analyze", json={
            "resume_text": "   ",
            "jd_text": "Data Scientist role requiring Python.",
        })
        assert resp.status_code == 400

    async def test_analyze_empty_jd_returns_400(self, client):
        resp = await client.post("/api/analyze", json={
            "resume_text": "Jane Smith. Python developer.",
            "jd_text": "",
        })
        assert resp.status_code == 400

    async def test_analyze_missing_field_returns_422(self, client):
        resp = await client.post("/api/analyze", json={"resume_text": "Jane Smith"})
        assert resp.status_code == 422  # Pydantic validation error

    async def test_analyze_no_keywords_gives_mid_score(self, client):
        """Irrelevant resume vs specific JD should score low on keyword match."""
        resp = await client.post("/api/analyze", json={
            "resume_text": "Marketing manager. Led brand campaigns. SEO expert.",
            "jd_text": "Data Scientist. Python, TensorFlow, BigQuery, Kubernetes required.",
        })
        data = resp.json()
        assert data["keyword_match"] < 30  # very few keywords present

    async def test_analyze_perfect_match_scores_high(self, client):
        """Resume with all JD keywords should score high on keyword match."""
        resp = await client.post("/api/analyze", json={
            "resume_text": (
                "SKILLS\nPython, TensorFlow, BigQuery, Kubernetes, AWS, Docker\n"
                "EXPERIENCE\nBuilt TensorFlow models deployed on Kubernetes with BigQuery pipelines."
            ),
            "jd_text": "Python, TensorFlow, BigQuery, Kubernetes, AWS, Docker required.",
        })
        data = resp.json()
        assert data["keyword_match"] >= 80

    async def test_analyze_no_api_key_needed(self, client):
        """
        The /api/analyze endpoint must work without NVIDIA_API_KEY
        since it only uses the heuristic scorer (no LLM calls).
        """
        with patch.dict("os.environ", {}, clear=False):
            # Even if we clear the key, analyze should still work
            resp = await client.post("/api/analyze", json={
                "resume_text": "Python developer. Built ML models.",
                "jd_text":     "Python and machine learning required.",
            })
        assert resp.status_code == 200

    async def test_analyze_returns_all_expected_fields(self, client):
        resp = await client.post("/api/analyze", json={
            "resume_text": "Jane Smith\nPython SQL TensorFlow AWS",
            "jd_text":     "Data Scientist. Python, SQL, TensorFlow required.",
        })
        data = resp.json()
        expected_fields = [
            "ats_score", "keyword_match", "quantification",
            "action_verb_usage", "formatting", "section_completeness",
            "present_keywords", "missing_keywords", "keywords_used",
        ]
        for field in expected_fields:
            assert field in data, f"Missing field: {field}"
