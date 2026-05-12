"""
api/main.py
───────────────────────────────────────
FastAPI application for the AI Job Application Agent.

Endpoints:
  POST /api/jobs                         — Create session
  POST /api/jobs/{id}/upload             — Upload resume + JD files
  POST /api/jobs/{id}/upload-url         — Provide JD by URL
  POST /api/jobs/{id}/run                — Run full pipeline (async background)
  GET  /api/jobs/{id}/status             — Poll pipeline status
  GET  /api/jobs/{id}/result             — Get full results
  GET  /api/jobs/{id}/download/{type}    — Download resume/cover_letter DOCX
  GET  /api/jobs                         — List all sessions
  GET  /api/health                       — Health check
"""
from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional

from fastapi import BackgroundTasks, FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

from core.config import get_settings
from core.models import JobSession, JobStatus

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

cfg = get_settings()

# ─── Thread-safe session store ────────────────────────────────────────────────
# A simple dict protected by a lock so background threads and async handlers
# don't race when reading / writing session state.

_sessions: Dict[str, JobSession] = {}
_sessions_lock = threading.Lock()
_executor = ThreadPoolExecutor(max_workers=4)


def _get_session_safe(session_id: str) -> JobSession:
    with _sessions_lock:
        if session_id not in _sessions:
            raise HTTPException(404, f"Session '{session_id}' not found.")
        return _sessions[session_id]


def _set_session_safe(session_id: str, session: JobSession):
    with _sessions_lock:
        _sessions[session_id] = session


# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="AI Job Application Agent",
    description=(
        "Multi-agent pipeline for ATS-optimised resume tailoring and cover letter generation.\n\n"
        "**Powered by NVIDIA NIM (free tier)**\n\n"
        "### Typical workflow\n"
        "1. `POST /api/jobs` — create a session\n"
        "2. `POST /api/jobs/{id}/upload` — upload resume + JD files\n"
        "3. `POST /api/jobs/{id}/run` — start the pipeline (async background)\n"
        "4. `GET  /api/jobs/{id}/status` — poll until `status == complete`\n"
        "5. `GET  /api/jobs/{id}/result` — retrieve scores, resume, cover letter\n"
        "6. `GET  /api/jobs/{id}/download/resume` — download DOCX\n\n"
        "### Quick analysis (no pipeline)\n"
        "`POST /api/analyze` — score and gap-analyse a resume against a JD in one call (no rewriting)."
    ),
    version="2.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

Path(cfg.upload_dir).mkdir(parents=True, exist_ok=True)
Path(cfg.output_dir).mkdir(parents=True, exist_ok=True)


# ─── Request / Response models ────────────────────────────────────────────────

class CreateJobResponse(BaseModel):
    session_id: str
    message: str

class UploadURLRequest(BaseModel):
    jd_url: str

class StatusResponse(BaseModel):
    session_id: str
    status: str
    logs: List[str]
    error: Optional[str] = None

class ResultResponse(BaseModel):
    session_id: str
    status: str
    job_title: Optional[str] = None
    company: Optional[str] = None
    ats_score_before: Optional[float] = None
    ats_score_after: Optional[float] = None
    match_score: Optional[float] = None
    keywords_added: List[str] = []
    cover_letter: Optional[str] = None
    resume_text: Optional[str] = None
    ready_to_submit: Optional[bool] = None
    submission_notes: Optional[str] = None
    strengths: List[str] = []
    priority_rewrites: List[str] = []


class AnalyzeRequest(BaseModel):
    resume_text: str
    jd_text: str
    top_keywords: int = 20


class AnalyzeResponse(BaseModel):
    """Quick analysis result — no rewriting, no API calls to the LLM."""
    ats_score: float
    keyword_match: float
    quantification: float
    action_verb_usage: float
    formatting: float
    section_completeness: float
    present_keywords: List[str]
    missing_keywords: List[str]
    keywords_used: List[str]


# ─── Endpoints ────────────────────────────────────────────────────────────────

@app.post("/api/jobs", response_model=CreateJobResponse)
async def create_job():
    """Create a new job application session."""
    session_id = str(uuid.uuid4())
    _set_session_safe(session_id, JobSession(id=session_id))
    return CreateJobResponse(session_id=session_id, message="Session created. Upload files next.")


@app.post("/api/jobs/{session_id}/upload")
async def upload_files(
    session_id: str,
    resume: UploadFile = File(..., description="Resume (PDF or DOCX)"),
    jd: Optional[UploadFile] = File(None, description="Job Description (PDF or TXT)"),
):
    """Upload resume and optional JD file."""
    session = _get_session_safe(session_id)

    resume_path = Path(cfg.upload_dir) / f"{session_id}_resume{Path(resume.filename).suffix}"
    resume_path.write_bytes(await resume.read())
    session.resume_path = str(resume_path)
    session.log(f"Resume uploaded: {resume.filename}")

    if jd:
        jd_path = Path(cfg.upload_dir) / f"{session_id}_jd{Path(jd.filename).suffix}"
        jd_path.write_bytes(await jd.read())
        session.jd_path = str(jd_path)
        session.log(f"JD uploaded: {jd.filename}")

    _set_session_safe(session_id, session)
    return {"message": "Files uploaded.", "resume": resume.filename, "jd": jd.filename if jd else None}


@app.post("/api/jobs/{session_id}/upload-url")
async def upload_jd_url(session_id: str, request: UploadURLRequest):
    """Provide JD by URL (scraped automatically when pipeline runs)."""
    session = _get_session_safe(session_id)
    session.jd_url = request.jd_url
    session.log(f"JD URL set: {request.jd_url}")
    _set_session_safe(session_id, session)
    return {"message": "JD URL saved.", "url": request.jd_url}


@app.post("/api/jobs/{session_id}/run")
async def run_pipeline(session_id: str, background_tasks: BackgroundTasks):
    """Trigger the full agent pipeline in the background."""
    session = _get_session_safe(session_id)

    if session.status not in (JobStatus.PENDING, JobStatus.FAILED):
        raise HTTPException(400, f"Session is already in state: {session.status}")
    if not session.resume_path:
        raise HTTPException(400, "No resume uploaded. POST to /upload first.")
    if not session.jd_path and not session.jd_url:
        raise HTTPException(400, "No JD provided. Upload a file or provide a URL.")

    # Mark as SCRAPING immediately so concurrent /run calls are rejected
    # before the background thread has a chance to update the status itself.
    session.status = JobStatus.SCRAPING
    _set_session_safe(session_id, session)

    if session.jd_url and not session.jd_path:
        background_tasks.add_task(_run_with_url_jd, session_id)
    else:
        background_tasks.add_task(_run_pipeline, session_id)

    return {"message": "Pipeline started.", "session_id": session_id}


@app.get("/api/jobs/{session_id}/status", response_model=StatusResponse)
async def get_status(session_id: str):
    """Poll pipeline progress."""
    session = _get_session_safe(session_id)
    return StatusResponse(
        session_id=session_id,
        status=session.status.value,
        logs=session.agent_logs[-20:],
        error=session.error,
    )


@app.get("/api/jobs/{session_id}/result", response_model=ResultResponse)
async def get_result(session_id: str):
    """Get full pipeline results."""
    session = _get_session_safe(session_id)

    if session.status != JobStatus.COMPLETE:
        raise HTTPException(400, f"Pipeline not complete yet. Status: {session.status}")

    pkg = session.package
    analysis = session.analysis

    return ResultResponse(
        session_id=session_id,
        status=session.status.value,
        job_title=session.extracted_jd.job_title if session.extracted_jd else None,
        company=session.extracted_jd.company if session.extracted_jd else None,
        ats_score_before=analysis.ats_score.overall if analysis else None,
        ats_score_after=pkg.ats_score_after if pkg else None,
        match_score=analysis.match_score if analysis else None,
        keywords_added=pkg.keywords_added if pkg else [],
        cover_letter=pkg.cover_letter.body if pkg else None,
        resume_text=pkg.resume_text if pkg else None,
        ready_to_submit=pkg.ready_to_submit if pkg else None,
        submission_notes=pkg.submission_notes if pkg else None,
        strengths=analysis.strengths if analysis else [],
        priority_rewrites=analysis.priority_rewrites if analysis else [],
    )


@app.get("/api/jobs/{session_id}/download/{doc_type}")
async def download_file(session_id: str, doc_type: str):
    """Download resume or cover letter as DOCX."""
    session = _get_session_safe(session_id)
    pkg = session.package

    if not pkg:
        raise HTTPException(404, "No package generated yet.")

    if doc_type == "resume":
        path = pkg.resume_docx_path
        filename = f"resume_{session.extracted_jd.job_title if session.extracted_jd else 'tailored'}.docx"
    elif doc_type == "cover_letter":
        path = pkg.cover_letter_docx_path
        filename = f"cover_letter_{session.extracted_jd.job_title if session.extracted_jd else ''}.docx"
    else:
        raise HTTPException(400, "doc_type must be 'resume' or 'cover_letter'")

    if not path or not Path(path).exists():
        raise HTTPException(404, f"File not found: {path}")

    return FileResponse(path, filename=filename, media_type="application/octet-stream")


@app.get("/api/jobs", response_model=List[dict])
async def list_jobs():
    """List all sessions."""
    with _sessions_lock:
        snapshot = list(_sessions.values())
    return [
        {
            "id": s.id,
            "status": s.status.value,
            "created_at": s.created_at.isoformat(),
            "job_title": s.extracted_jd.job_title if s.extracted_jd else None,
            "company": s.extracted_jd.company if s.extracted_jd else None,
        }
        for s in snapshot
    ]


@app.post("/api/analyze", response_model=AnalyzeResponse)
async def quick_analyze(request: AnalyzeRequest):
    """
    Score and gap-analyse a resume against a JD instantly.

    Uses only the heuristic ATS scorer — no LLM calls, no API key needed.
    Returns in <50ms and works even when the NVIDIA NIM endpoint is degraded.

    Use this to:
    - Quickly check a resume's baseline score before running the full pipeline
    - Build a resume scoring widget without any LLM infrastructure
    - Benchmark the before/after improvement after manual edits
    """
    from tools.ats_scorer import extract_jd_keywords, score_resume

    if not request.resume_text.strip():
        raise HTTPException(400, "resume_text must not be empty.")
    if not request.jd_text.strip():
        raise HTTPException(400, "jd_text must not be empty.")

    keywords = extract_jd_keywords(request.jd_text, top_k=request.top_keywords)
    score    = score_resume(request.resume_text, keywords)

    return AnalyzeResponse(
        ats_score=score.overall,
        keyword_match=score.keyword_match,
        quantification=score.quantification,
        action_verb_usage=score.action_verb_usage,
        formatting=score.formatting,
        section_completeness=score.section_completeness,
        present_keywords=score.present_keywords,
        missing_keywords=score.missing_keywords,
        keywords_used=keywords,
    )


@app.get("/api/health")
async def health():
    """Health check endpoint."""
    with _sessions_lock:
        total = len(_sessions)
        by_status = {}
        for s in _sessions.values():
            by_status[s.status.value] = by_status.get(s.status.value, 0) + 1
    return {"status": "ok", "sessions": total, "by_status": by_status}


# ─── Background tasks ─────────────────────────────────────────────────────────

def _run_pipeline(session_id: str):
    """Blocking pipeline run — called from a thread-pool worker."""
    from agents.orchestrator import Orchestrator
    session = _get_session_safe(session_id)
    try:
        orchestrator = Orchestrator()
        result = orchestrator.run(session)
        _set_session_safe(session_id, result)
    except Exception as exc:
        logger.exception(f"Pipeline worker crashed for session {session_id}: {exc}")
        session.status = JobStatus.FAILED
        session.error  = str(exc)
        _set_session_safe(session_id, session)


async def _run_with_url_jd(session_id: str):
    """Scrape JD from URL, then hand off to the synchronous pipeline."""
    from parsers.pdf_parser import extract_from_url

    session = _get_session_safe(session_id)
    try:
        jd_text = await extract_from_url(session.jd_url)
        jd_path = Path(cfg.upload_dir) / f"{session_id}_jd.txt"
        jd_path.write_text(jd_text, encoding="utf-8")
        session.jd_path = str(jd_path)
        session.log(f"JD scraped from URL: {len(jd_text)} chars")
        _set_session_safe(session_id, session)
    except Exception as e:
        session.status = JobStatus.FAILED
        session.error = f"Failed to scrape JD URL: {e}"
        _set_session_safe(session_id, session)
        return

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(_executor, _run_pipeline, session_id)


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api.main:app", host=cfg.api_host, port=cfg.api_port, reload=True)
