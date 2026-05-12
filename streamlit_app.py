"""
streamlit_app.py
─────────────────
Streamlit UI for the AI Job Application Agent.

Run with:
    streamlit run streamlit_app.py

API key is loaded exclusively from the .env file — never entered in the UI.
See .env.example for setup instructions.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import os
import sys
import tempfile
import time
import uuid
from pathlib import Path
from typing import Optional

import pandas as pd
import streamlit as st
from dotenv import load_dotenv

# Load .env before anything else
load_dotenv()

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).parent))

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Job Application Agent",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom theme & styles ────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global font & background ── */
html, body, [class*="css"] {
    font-family: 'Inter', 'Segoe UI', sans-serif;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a 0%, #1e293b 100%);
    border-right: 1px solid #334155;
}
[data-testid="stSidebar"] * {
    color: #e2e8f0 !important;
}

/* ── Main background ── */
[data-testid="stAppViewContainer"] {
    background-color: #0f172a;
}
[data-testid="stMain"] {
    background-color: #0f172a;
}

/* ── Headers ── */
h1 { color: #f8fafc !important; font-weight: 700 !important; }
h2 { color: #e2e8f0 !important; font-weight: 600 !important; }
h3 { color: #cbd5e1 !important; font-weight: 600 !important; }
p, li, span { color: #94a3b8; }

/* ── Primary button (Run Pipeline) ── */
[data-testid="stButton"] > button[kind="primary"] {
    background: linear-gradient(135deg, #6366f1 0%, #8b5cf6 100%);
    border: none;
    color: white;
    font-weight: 600;
    font-size: 1rem;
    border-radius: 10px;
    transition: all 0.2s ease;
    box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
}
[data-testid="stButton"] > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(99, 102, 241, 0.6);
}

/* ── Download buttons ── */
[data-testid="stDownloadButton"] > button {
    background: #1e293b;
    border: 2px solid #6366f1;
    color: #e2e8f0 !important;
    font-weight: 600;
    border-radius: 8px;
    transition: all 0.2s ease;
}
[data-testid="stDownloadButton"] > button[kind="primary"] {
    background: #1e293b;
    border: 2px solid #10b981;
    color: #6ee7b7 !important;
    font-weight: 700;
}
[data-testid="stDownloadButton"] > button:hover {
    background: #334155;
    transform: translateY(-1px);
}
[data-testid="stDownloadButton"] > button p,
[data-testid="stDownloadButton"] > button span {
    color: inherit !important;
}

/* ── Cards / containers ── */
[data-testid="stMetric"] {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1rem 1.2rem;
}
[data-testid="stMetricValue"] { color: #f8fafc !important; font-weight: 700; }
[data-testid="stMetricDelta"] { font-weight: 600; }

/* ── Tabs ── */
[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: #1e293b;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #334155;
}
[data-testid="stTabs"] [data-baseweb="tab"] {
    border-radius: 8px;
    color: #94a3b8 !important;
    font-weight: 500;
}
[data-testid="stTabs"] [aria-selected="true"] {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
}

/* ── Alerts / banners ── */
[data-testid="stSuccess"] {
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(16, 185, 129, 0.4);
    border-radius: 10px;
    color: #6ee7b7 !important;
}
[data-testid="stWarning"] {
    background: rgba(245, 158, 11, 0.12);
    border: 1px solid rgba(245, 158, 11, 0.4);
    border-radius: 10px;
    color: #fcd34d !important;
}
[data-testid="stError"] {
    background: rgba(239, 68, 68, 0.12);
    border: 1px solid rgba(239, 68, 68, 0.4);
    border-radius: 10px;
}
[data-testid="stInfo"] {
    background: rgba(99, 102, 241, 0.12);
    border: 1px solid rgba(99, 102, 241, 0.4);
    border-radius: 10px;
    color: #a5b4fc !important;
}

/* ── Text areas / inputs ── */
textarea, input[type="text"] {
    background: #1e293b !important;
    border: 1px solid #334155 !important;
    border-radius: 8px !important;
    color: #e2e8f0 !important;
}

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #1e293b;
    border: 2px dashed #334155;
    border-radius: 12px;
    padding: 1rem;
}

/* ── Divider ── */
hr { border-color: #1e293b !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #334155; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #6366f1; }
</style>
""", unsafe_allow_html=True)


# ─── Session state init ───────────────────────────────────────────────────────

def _init_state():
    for k, v in {
        "job_session": None,
        "pipeline_done": False,
        "error": None,
        "tmp_dir": None,
        "pipeline_future": None,
        "pipeline_start_time": None,
    }.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init_state()


# ─── Load settings from .env ──────────────────────────────────────────────────

def _load_cfg():
    """Return Settings, or None with an error string if misconfigured."""
    try:
        from core.config import get_settings
        get_settings.cache_clear()
        return get_settings(), None
    except Exception as exc:
        return None, str(exc)

cfg, cfg_error = _load_cfg()
api_ready = cfg is not None and bool(cfg.nvidia_api_key)


# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("💼 Job Agent")
    st.caption("Powered by NVIDIA NIM")
    st.divider()

    # ── Pipeline settings ─────────────────────────────────────
    st.subheader("⚙️ Pipeline Settings")

    model_choice = st.selectbox(
        "Model",
        [
            "meta/llama-3.1-8b-instruct",
            "meta/llama-3.1-70b-instruct",
            "meta/llama-3.3-70b-instruct",
            "nvidia/llama-3.1-nemotron-70b-instruct",
        ],
        index=0,
        help="8B = fast (~2 min). 70B = higher quality but slow (~15 min). Recommended: 8B.",
    )
    if "70b" in model_choice or "nemotron" in model_choice:
        st.caption("⚠️ 70B models can take 10–20 min on the free tier.")

    target_ats = st.slider(
        "Target ATS Score",
        min_value=50, max_value=95, value=75, step=5,
        help="Pipeline keeps rewriting until this score is reached.",
    )
    iterations = st.radio(
        "Max Rewrite Iterations",
        [1, 2, 3],
        index=0,
        horizontal=True,
        help="1 = fastest (~2 min). More iterations improve quality but add time.",
    )

    st.divider()

    # ── Personal style learning ───────────────────────────────
    with st.expander("🖊️ My Writing Style (optional)"):
        st.caption(
            "Paste 3–5 of your best resume bullets. "
            "The agent will mirror your authentic voice when rewriting."
        )
        style_input = st.text_area(
            "Your best bullets",
            height=140,
            placeholder=(
                "• Led migration of 3 legacy services to microservices, cutting p99 latency by 40%\n"
                "• Automated CI/CD pipeline for 8 services; reduced deploy time from 45 min to 6 min\n"
                "• Built real-time anomaly detection model (F1=0.94) deployed to 12M users"
            ),
            key="style_bullets_input",
            label_visibility="collapsed",
        )
        if st.button("💾 Save Style Samples", key="save_style_btn", use_container_width=True):
            if style_input.strip():
                try:
                    from core.config import get_settings
                    from core.memory import AgentMemory
                    cfg2 = get_settings()
                    mem = AgentMemory(persist_dir=cfg2.chroma_dir)
                    bullets = [
                        ln.lstrip("•-– ").strip()
                        for ln in style_input.splitlines()
                        if ln.strip()
                    ]
                    mem.store_style_samples(bullets)
                    st.success(f"✅ Saved {len(bullets)} style sample(s).")
                except Exception as exc:
                    st.error(f"Could not save: {exc}")
            else:
                st.warning("Please paste at least one bullet point.")

    st.divider()
    if st.button("🔄 Reset", use_container_width=True, help="Clear results and start over"):
        for k in ["job_session", "pipeline_done", "error", "tmp_dir"]:
            st.session_state[k] = None if k != "pipeline_done" else False
        st.rerun()


# ─── Header ───────────────────────────────────────────────────────────────────

st.title("💼 AI Job Application Agent")
st.caption(
    "Upload your resume and a job description — the multi-agent pipeline will "
    "analyse the gap, tailor your resume for ATS, and write a cover letter."
)

if not api_ready:
    st.error(
        "**NVIDIA_API_KEY is not configured.** "
        "Open `.env`, add `NVIDIA_API_KEY=nvapi-...`, then restart Streamlit. "
        "Get a free key at https://build.nvidia.com"
    )
    st.stop()

st.divider()


# ─── Step 1: Upload ───────────────────────────────────────────────────────────

st.header("Step 1 — Upload Documents")

col_resume, col_jd = st.columns(2, gap="large")

with col_resume:
    st.subheader("📄 Your Resume")
    resume_file = st.file_uploader(
        "Upload PDF or DOCX",
        type=["pdf", "docx"],
        key="resume_upload",
        label_visibility="collapsed",
    )
    if resume_file:
        st.success(f"✓  {resume_file.name}  ({resume_file.size // 1024} KB)")

with col_jd:
    st.subheader("📋 Job Description")
    jd_tab_file, jd_tab_url, jd_tab_paste = st.tabs(["File", "URL", "Paste"])

    jd_file: Optional[object] = None
    jd_url: str = ""
    jd_paste: str = ""

    with jd_tab_file:
        jd_file = st.file_uploader(
            "Upload PDF or TXT",
            type=["pdf", "txt"],
            key="jd_upload",
            label_visibility="collapsed",
        )
        if jd_file:
            st.success(f"✓  {jd_file.name}")

    with jd_tab_url:
        jd_url = st.text_input(
            "Job posting URL",
            placeholder="https://www.linkedin.com/jobs/view/...",
            label_visibility="collapsed",
        )
        if jd_url:
            st.info(f"Will scrape: {jd_url[:60]}...")

    with jd_tab_paste:
        jd_paste = st.text_area(
            "Paste the full job description here",
            height=220,
            placeholder="Senior Software Engineer\n\nAbout the role...",
            label_visibility="collapsed",
        )

st.divider()


# ─── Step 2: Run ──────────────────────────────────────────────────────────────

st.header("Step 2 — Run Pipeline")

has_jd = bool(jd_file or jd_url or jd_paste)
can_run = bool(resume_file and has_jd)

if not can_run:
    missing = []
    if not resume_file:
        missing.append("resume file")
    if not has_jd:
        missing.append("job description")
    st.info(f"**Still needed:** {' · '.join(missing)}")

run_clicked = st.button(
    "🚀  Run AI Pipeline",
    disabled=not can_run,
    use_container_width=True,
    type="primary",
)


def _keyword_badges(keywords: list, color: str = "blue", max_kw: int = 20) -> None:
    """
    Render a list of keyword strings as uniform HTML pill badges.

    Using HTML instead of markdown code-spans (`kw`) avoids broken rendering
    when the LLM returns keywords that contain stray backtick characters.

    color: "blue"  -> indigo / missing keywords style
           "green" -> emerald / present/added keywords style
           "amber" -> for warnings
    """
    palette = {
        "blue":  ("background:#1e3a5f; color:#93c5fd; border:1px solid #3b82f6;"),
        "green": ("background:#14532d; color:#86efac; border:1px solid #22c55e;"),
        "amber": ("background:#451a03; color:#fcd34d; border:1px solid #f59e0b;"),
    }
    style = palette.get(color, palette["blue"])
    base = (
        "display:inline-block; border-radius:6px; padding:3px 10px;"
        "margin:3px 4px; font-size:0.82em; font-weight:500;"
    )
    # Sanitise: strip backticks, asterisks, and leading/trailing punctuation
    # that the LLM sometimes adds around keywords.
    clean = [
        kw.strip().replace("`", "").replace("*", "").strip(" '\".,;:-")
        for kw in keywords
        if kw.strip().replace("`", "").strip()
    ][:max_kw]

    if not clean:
        return
    badges = "".join(f'<span style="{base}{style}">{kw}</span>' for kw in clean)
    st.markdown(f'<div style="line-height:2.4">{badges}</div>', unsafe_allow_html=True)


def _run_async_in_thread(coro_factory):
    """
    Run an async coroutine safely from Streamlit's sync context.
    Pass a zero-arg lambda, NOT the coroutine directly:
        _run_async_in_thread(lambda: extract_from_url(url))
    """
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro_factory())
        return future.result()


def _pipeline_worker(cfg_dict: dict):
    """
    Runs the full pipeline in a background thread.
    Receives config as a plain dict to avoid Streamlit serialization issues.
    Returns a completed JobSession.
    """
    import os
    os.environ["LLM_MODEL"]          = cfg_dict["model"]
    os.environ["TARGET_ATS_SCORE"]   = str(cfg_dict["target_ats"])
    os.environ["REWRITE_ITERATIONS"] = str(cfg_dict["iterations"])
    os.environ["OUTPUT_DIR"]         = cfg_dict["output_dir"]

    from core.config import get_settings
    get_settings.cache_clear()

    from agents.orchestrator import Orchestrator
    from core.models import JobSession

    session = JobSession(
        id=cfg_dict["session_id"],
        resume_path=cfg_dict["resume_path"],
        jd_path=cfg_dict["jd_path"],
    )
    orch = Orchestrator()
    orch.app_agent.output_dir = cfg_dict["output_dir"]
    return orch.run(session)


if run_clicked and can_run:
    # Push pipeline settings into env so get_settings() picks them up
    os.environ["LLM_MODEL"] = model_choice
    os.environ["TARGET_ATS_SCORE"] = str(target_ats)
    os.environ["REWRITE_ITERATIONS"] = str(iterations)

    # Fresh temp workspace for uploads / output
    tmp = Path(tempfile.mkdtemp(prefix="job_agent_"))
    uploads = tmp / "uploads"
    output  = tmp / "output"
    uploads.mkdir()
    output.mkdir()
    os.environ["UPLOAD_DIR"] = str(uploads)
    os.environ["OUTPUT_DIR"] = str(output)
    st.session_state.tmp_dir = str(tmp)

    # Re-load settings with updated env vars
    from core.config import get_settings
    get_settings.cache_clear()

    from core.models import JobSession

    session = JobSession(id=str(uuid.uuid4()))

    # ── Save resume ───────────────────────────────────────────
    resume_path = uploads / f"resume{Path(resume_file.name).suffix}"
    resume_path.write_bytes(resume_file.read())
    session.resume_path = str(resume_path)

    # ── Save JD ───────────────────────────────────────────────
    if jd_file:
        jd_path = uploads / f"jd{Path(jd_file.name).suffix}"
        jd_path.write_bytes(jd_file.read())
        session.jd_path = str(jd_path)

    elif jd_url:
        from parsers.pdf_parser import extract_from_url
        with st.spinner("🌐 Scraping job posting from URL…"):
            try:
                scraped = _run_async_in_thread(lambda: extract_from_url(jd_url))
                jd_path = uploads / "jd_scraped.txt"
                jd_path.write_text(scraped, encoding="utf-8")
                session.jd_path = str(jd_path)
                st.success(f"Scraped {len(scraped):,} characters.")
            except Exception as exc:
                st.error(f"Failed to scrape URL: {exc}")
                st.stop()

    else:
        jd_path = uploads / "jd_pasted.txt"
        jd_path.write_text(jd_paste, encoding="utf-8")
        session.jd_path = str(jd_path)

    # ── Launch pipeline in background thread (non-blocking) ──
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    future = executor.submit(_pipeline_worker, {
        "session_id":  str(session.id),
        "resume_path": str(session.resume_path),
        "jd_path":     str(session.jd_path),
        "model":       model_choice,
        "target_ats":  target_ats,
        "iterations":  iterations,
        "output_dir":  str(output),
    })
    st.session_state.pipeline_future     = future
    st.session_state.pipeline_start_time = time.time()
    st.session_state.pipeline_done       = False
    st.rerun()

# ── Poll for pipeline completion ──────────────────────────
if st.session_state.get("pipeline_future") is not None:
    future    = st.session_state.pipeline_future
    elapsed   = int(time.time() - (st.session_state.pipeline_start_time or time.time()))
    mins, sec = divmod(elapsed, 60)

    if future.done():
        try:
            completed_session = future.result()
            st.session_state.job_session     = completed_session
            st.session_state.pipeline_done   = True
            st.session_state.pipeline_future = None
            session = completed_session
        except Exception as exc:
            st.session_state.error           = str(exc)
            st.session_state.pipeline_done   = True
            st.session_state.pipeline_future = None
            # Build a minimal failed session for display
            from core.models import JobSession as _JS
            from core.models import JobStatus as _JSt
            session = _JS(id="error")
            session.status = _JSt.FAILED
            session.error  = str(exc)
            st.session_state.job_session = session
        st.rerun()
    else:
        # Still running — show live progress and poll every 4 s
        st.info(
            f"🤖 Pipeline running with **{st.session_state.get('_model', model_choice)}** "
            f"— {mins}m {sec:02d}s elapsed…  \n"
            "This typically takes **1–3 minutes** on the 8B model."
        )
        progress_phases = [
            (20,  "📄 Phase 1: Parsing documents…"),
            (40,  "🔍 Phase 2: Analyzing skill gaps…"),
            (70,  "✍️ Phase 3: Rewriting resume…"),
            (90,  "📦 Phase 4: Assembling package…"),
            (100, "✅ Finishing up…"),
        ]
        # Approximate progress based on elapsed time (total ~120 s for 8B)
        approx_pct = min(95, int(elapsed / 120 * 100))
        label = next((lbl for pct, lbl in progress_phases if approx_pct <= pct),
                     progress_phases[-1][1])
        st.progress(approx_pct / 100, text=label)
        time.sleep(4)
        st.rerun()

if st.session_state.pipeline_done and st.session_state.job_session:
    session = st.session_state.job_session

    if session.status.value == "complete":
        warnings = [ln for ln in session.agent_logs if "WARNING" in ln]
        if warnings:
            st.success("✅ Pipeline complete (with warnings — see Agent log)")
        else:
            st.success("✅ Pipeline complete!")
    elif session.status.value == "failed":
        err = str(session.error or st.session_state.error or "")
        err_lower = err.lower()
        if "degraded" in err_lower or ("400" in err and "bad request" in err_lower):
            st.error("❌ NVIDIA NIM: the selected model endpoint is **DEGRADED** (temporarily offline).")
            st.warning(
                "**What to do:**\n"
                "1. Open the sidebar and switch to a **different model** — "
                "try `meta/llama-3.1-70b-instruct` or `nvidia/llama-3.1-nemotron-70b-instruct`.\n"
                "2. Or wait 10–15 minutes and try again with the same model.\n"
                "3. Check model status at https://build.nvidia.com"
            )
        elif "rate" in err_lower or "429" in err or "quota" in err_lower:
            st.error("❌ NVIDIA NIM rate limit hit.")
            st.warning(
                "**What to do:**\n"
                "1. Wait 60 seconds and click **Run** again.\n"
                "2. Switch to a smaller model (`llama-3.1-8b-instruct`) in the sidebar.\n"
                "3. Get a second free key at https://build.nvidia.com"
            )
        else:
            st.error(f"❌ Pipeline failed at an early stage: {err}")
    else:
        st.warning("⚠️ Pipeline finished with partial results — see Agent log for details.")

    with st.expander("📜 Agent log", expanded=False):
        for line in session.agent_logs:
            st.text(line)


# ─── Step 3: Results ──────────────────────────────────────────────────────────

if st.session_state.pipeline_done and st.session_state.job_session:
    session = st.session_state.job_session
    st.divider()
    st.header("Step 3 — Results")

    # Show whatever we have — even a partially-failed pipeline may have
    # completed Phase 1-3 (analysis + rewritten resume) successfully.
    has_resume  = bool(session.rewritten_resume and session.rewritten_resume.full_text)
    has_package = bool(session.package)

    if session.status.value == "failed" and not has_resume:
        # Truly nothing to show
        st.error(f"Pipeline failed before producing results: {session.error}")
        st.stop()
    elif session.status.value == "failed":
        st.warning(
            f"⚠️ Pipeline failed at a late stage but partial results are available below. "
            f"Error: {session.error}"
        )

    # ── Key metrics ───────────────────────────────────────────
    mc1, mc2, mc3, mc4 = st.columns(4)
    with mc1:
        if session.analysis:
            st.metric("Match Score", f"{session.analysis.match_score:.0f} / 100")
    with mc2:
        if session.analysis:
            st.metric("ATS Before", f"{session.analysis.ats_score.overall:.0f} / 100")
    with mc3:
        if session.rewritten_resume and session.rewritten_resume.new_ats_score:
            new = session.rewritten_resume.new_ats_score.overall
            old = session.analysis.ats_score.overall
            st.metric("ATS After", f"{new:.0f} / 100", delta=f"{new - old:+.1f} pts")
    with mc4:
        if session.package:
            st.metric("Submission", "✅ Ready" if session.package.ready_to_submit else "⚠️ Review needed")

    st.divider()

    # ── Quick Download Bar (always visible) ───────────────────
    st.subheader("⬇️ Download Your Files")
    qd1, qd2 = st.columns(2)

    _resume_text = (
        session.package.resume_text if has_package and session.package.resume_text
        else session.rewritten_resume.full_text if has_resume
        else None
    )
    _resume_docx = session.package.resume_docx_path if has_package else None
    _cl_obj      = session.package.cover_letter if has_package else None
    _cl_docx     = session.package.cover_letter_docx_path if has_package else None

    with qd1:
        if _resume_docx and Path(_resume_docx).exists():
            with open(_resume_docx, "rb") as _fh:
                st.download_button(
                    "⬇️  Tailored Resume (DOCX)", _fh.read(),
                    file_name="tailored_resume.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True, type="primary",
                    key="dl_resume_docx_top",
                )
        elif _resume_text:
            st.download_button(
                "⬇️  Tailored Resume (TXT)", _resume_text,
                file_name="tailored_resume.txt", mime="text/plain",
                use_container_width=True, type="primary",
                key="dl_resume_txt_top",
            )
        else:
            st.info("Resume file not ready.")

    with qd2:
        _cl_ok = _cl_obj and _cl_obj.body and "could not be generated" not in _cl_obj.body
        if _cl_docx and Path(_cl_docx).exists():
            with open(_cl_docx, "rb") as _fh:
                st.download_button(
                    "⬇️  Cover Letter (DOCX)", _fh.read(),
                    file_name="cover_letter.docx",
                    mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document",
                    use_container_width=True,
                    key="dl_cl_docx_top",
                )
        elif _cl_ok:
            _cl_txt = f"{_cl_obj.subject_line}\n\n{_cl_obj.body}"
            st.download_button(
                "⬇️  Cover Letter (TXT)", _cl_txt,
                file_name="cover_letter.txt", mime="text/plain",
                use_container_width=True,
                key="dl_cl_txt_top",
            )
        else:
            st.info("Cover letter not ready.")

    st.divider()

    # ── Result tabs ───────────────────────────────────────────
    tab_analysis, tab_resume, tab_cover = st.tabs([
        "📊 Analysis", "✍️ Rewritten Resume", "📝 Cover Letter"
    ])

    # ── Analysis ──────────────────────────────────────────────
    with tab_analysis:
        if session.extracted_jd:
            st.subheader(f"{session.extracted_jd.job_title} at {session.extracted_jd.company}")

        if session.analysis:
            st.subheader("ATS Score Breakdown")
            ats = session.analysis.ats_score
            ats_df = pd.DataFrame({
                "Dimension": ["Keyword Match", "Formatting", "Section Completeness",
                               "Quantification", "Action Verbs"],
                "Score": [ats.keyword_match, ats.formatting, ats.section_completeness,
                          ats.quantification, ats.action_verb_usage],
            })
            st.bar_chart(ats_df.set_index("Dimension"), height=250)

            two_cols = st.columns(2)
            with two_cols[0]:
                st.subheader("✅ Strengths")
                for s in session.analysis.strengths:
                    st.success(s)
            with two_cols[1]:
                st.subheader("🎯 Priority Rewrites")
                for r in session.analysis.priority_rewrites[:6]:
                    st.warning(r)

            st.subheader("Skill Gap Detail")
            if session.analysis.skill_gaps:
                rows = [{
                    "Requirement": g.requirement[:70],
                    "Match": g.match.value,
                    "Importance": g.importance,
                    "Suggestion": (g.rewrite_suggestion or "—")[:90],
                } for g in session.analysis.skill_gaps]

                def _color(val):
                    return {
                        "missing": "background-color:#7f1d1d; color:#fecaca; font-weight:600",
                        "partial":  "background-color:#78350f; color:#fde68a; font-weight:600",
                        "strong":   "background-color:#14532d; color:#86efac; font-weight:600",
                    }.get(val, "")

                st.dataframe(
                    pd.DataFrame(rows).style.map(_color, subset=["Match"]),
                    use_container_width=True,
                    hide_index=True,
                )

            if ats.missing_keywords:
                st.subheader("🔑 Missing Keywords")
                _keyword_badges(ats.missing_keywords, color="blue")

        # ── Critique Report ───────────────────────────────────────
        if session.critique:
            cr = session.critique
            st.divider()
            st.subheader("🔍 AI Self-Critique Report")
            verdict_col, kw_col, star_col = st.columns(3)
            with verdict_col:
                verdict = "✅ Passed" if cr.passed else "❌ Did not pass"
                delta_color = "normal" if cr.passed else "inverse"
                st.metric("Verdict", verdict, f"{cr.overall_score:.0f}/100")
            with kw_col:
                st.metric("Keyword Coverage", f"{cr.keyword_coverage:.0f}%")
            with star_col:
                st.metric("STAR Compliance", f"{cr.star_compliance:.0f}%")

            if cr.praise:
                st.markdown("**👍 What's working well**")
                for p in cr.praise:
                    st.success(p)

            if cr.issues:
                st.markdown("**⚠️ Issues found**")
                issue_rows = [{
                    "Section": i.section,
                    "Issue": i.issue[:80],
                    "Severity": i.severity,
                    "Fix": i.suggestion[:80],
                } for i in cr.issues]

                def _sev_color(val):
                    return {
                        "critical": "background-color:#7f1d1d; color:#fecaca; font-weight:700",
                        "major":    "background-color:#78350f; color:#fde68a; font-weight:600",
                        "minor":    "background-color:#1e3a5f; color:#bae6fd; font-weight:500",
                    }.get(val, "")

                st.dataframe(
                    pd.DataFrame(issue_rows).style.map(_sev_color, subset=["Severity"]),
                    use_container_width=True,
                    hide_index=True,
                )

            if cr.fix_instructions and not cr.passed:
                with st.expander("📋 Fix instructions used for second rewrite pass"):
                    st.info(cr.fix_instructions)

    # ── Rewritten resume ──────────────────────────────────────
    with tab_resume:
        if session.rewritten_resume:
            if session.rewritten_resume.added_keywords:
                st.markdown("**Keywords added to resume:**")
                _keyword_badges(session.rewritten_resume.added_keywords, color="green")
            st.text_area("Rewritten Resume", session.rewritten_resume.full_text,
                         height=550, key="resume_text_area")
            if session.rewritten_resume.sections:
                with st.expander("Section-by-section changes"):
                    for sec in session.rewritten_resume.sections:
                        st.markdown(f"**{sec.section}**")
                        c1, c2 = st.columns(2)
                        with c1:
                            st.caption("Before")
                            st.code(sec.original[:400], language=None)
                        with c2:
                            st.caption("After")
                            st.code(sec.rewritten[:400], language=None)
                        st.divider()

    # ── Cover letter ──────────────────────────────────────────
    with tab_cover:
        if session.package and session.package.cover_letter:
            cl = session.package.cover_letter
            st.subheader(cl.subject_line or "Cover Letter")
            st.caption(f"Tone: {cl.tone}")
            if "could not be generated" in cl.body or "failed" in cl.body.lower():
                st.warning("Cover letter could not be generated automatically. Please write it manually.")
            else:
                st.text_area("Cover Letter Body", cl.body, height=450, key="cover_text_area")
        else:
            st.info("Cover letter not yet generated.")

    # ── Agent log (below tabs) ────────────────────────────────────
    with st.expander("📄 Full agent log", expanded=False):
        for line in session.agent_logs:
            st.text(line)

    # ── Feedback loop ─────────────────────────────────────────────
    st.divider()
    st.subheader("📬 Track Your Application")
    st.caption(
        "After you apply, come back and record the outcome. "
        "The agent learns from your history to write better resumes over time."
    )
    with st.expander("Record application outcome", expanded=False):
        fb_col1, fb_col2 = st.columns(2)
        with fb_col1:
            fb_outcome = st.selectbox(
                "Outcome",
                ["got_interview", "got_offer", "rejected", "no_response"],
                format_func=lambda x: {
                    "got_interview": "🎉 Got an interview",
                    "got_offer":     "🏆 Got an offer",
                    "rejected":      "❌ Rejected",
                    "no_response":   "📭 No response",
                }[x],
                key="fb_outcome",
            )
        with fb_col2:
            fb_notes = st.text_input(
                "Notes (optional)",
                placeholder="e.g. Recruiter called, liked Python background",
                key="fb_notes",
            )

        if st.button("💾 Save Feedback", key="save_feedback_btn"):
            try:
                from core.config import get_settings
                from core.memory import AgentMemory
                cfg2 = get_settings()
                mem = AgentMemory(persist_dir=cfg2.chroma_dir)
                jt  = session.extracted_jd.job_title if session.extracted_jd else "Unknown"
                co  = session.extracted_jd.company   if session.extracted_jd else "Unknown"
                mem.store_feedback(
                    session_id=session.id,
                    job_title=jt,
                    company=co,
                    outcome=fb_outcome,
                    notes=fb_notes,
                )
                st.success(f"✅ Feedback saved! ({jt} at {co} → {fb_outcome})")
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")
