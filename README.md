# AI Job Application Agent

> **A production-ready, multi-agent AI system that rewrites your resume for any job, scores it against ATS filters, self-critiques the output, and generates a personalised cover letter — all in under 60 seconds.**

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?logo=streamlit&logoColor=white)](https://resume-agent-xqvdwdsnfnwfoq7wxhf8nt.streamlit.app/)
[![HuggingFace Space](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/Mechscientist26/resume-agent)
[![CI](https://github.com/Mondip-Mech/resume-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/Mondip-Mech/resume-agent/actions)
[![Tests](https://img.shields.io/badge/tests-133%20passed-brightgreen)](#running-tests)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-lightgrey)](LICENSE)

---

## What This Project Demonstrates

This is a **full-stack AI engineering project** built end-to-end — from LLM orchestration and vector memory to a REST API, containerised deployment, and automated CI/CD. It showcases:

| Skill Area | What's implemented |
|---|---|
| **Agentic AI / LLM Engineering** | 5-agent pipeline with orchestration, self-critique loop, and RAG memory |
| **ML System Design** | ATS scorer validated at 86.7% accuracy (Spearman rho = 0.871) against human labels |
| **Backend Engineering** | FastAPI REST API with 10 endpoints, async background jobs, Pydantic v2 models |
| **MLOps / DevOps** | Docker, docker-compose, GitHub Actions CI/CD with 5 automated pipeline stages |
| **Testing** | 133 tests across agents, API, memory, LLM client, and orchestrator — no API key needed |
| **Vector Search / RAG** | ChromaDB + sentence-transformers embeddings with TF-IDF fallback |

---

## Project Highlights

- **5 specialised AI agents** working in a coordinated pipeline — scraping, analysing, rewriting, self-critiquing, and assembling the final application package
- **ATS scorer validated at 86.7% band accuracy** against 15 hand-labelled resume/JD pairs (Spearman rho = 0.871)
- **Self-critique loop** — if the rewritten resume doesn't pass quality checks, the system automatically runs a second fix pass before returning results
- **133 automated tests** covering every layer: agents, API endpoints, LLM retry logic, vector memory, and the full pipeline — all run in ~15 seconds with no API key needed
- **Full CI/CD pipeline** — 5 GitHub Actions jobs (tests × 3 Python versions, ATS benchmark, lint, Docker build, HuggingFace deploy) gate every merge to main
- **Two interfaces** — Streamlit web UI for end users + FastAPI REST backend for developers
- **Production patterns** — smart retry logic (skips retrying on DEGRADED model errors), rate limiting, containerised with named Docker volumes, DOCX export

---

## How It Works

Upload a resume and paste a job description. The pipeline runs automatically:

```
1. ScraperAgent    → Extracts structured data from your resume and the job description
2. AnalyzerAgent   → Identifies skill gaps, ATS score, and seniority match
3. RewriterAgent   → Rewrites your resume with targeted keywords and stronger action verbs
4. CritiqueAgent   → Reviews the rewrite; triggers a second fix pass if quality isn't met
5. ApplicationAgent → Generates a tailored cover letter and exports DOCX files
```

Output: an ATS-optimised resume + cover letter ready to submit, with before/after scores and a gap analysis report.

---

## Tech Stack

| Layer | Technology |
|---|---|
| **LLM** | NVIDIA NIM — Llama 3.1 8B (free tier) |
| **Agent Framework** | Custom multi-agent orchestrator with session state |
| **Embeddings / RAG** | sentence-transformers `all-MiniLM-L6-v2` + ChromaDB |
| **API** | FastAPI + Uvicorn (async, 10 endpoints) |
| **UI** | Streamlit (non-blocking async pipeline) |
| **Data Validation** | Pydantic v2 with field validators |
| **Document Export** | python-docx (resume + cover letter as .docx) |
| **PDF Parsing** | PyMuPDF → pdfplumber fallback |
| **Containers** | Docker + docker-compose (shared model cache volume) |
| **CI/CD** | GitHub Actions → HuggingFace Spaces |
| **Testing** | pytest + pytest-asyncio + httpx ASGITransport |

---

## Live Demo

Try it instantly — no setup, no API key needed:

**[Open Streamlit App](https://resume-agent-xqvdwdsnfnwfoq7wxhf8nt.streamlit.app/)** | **[Open HuggingFace Space](https://huggingface.co/spaces/Mechscientist26/resume-agent)**

---

## Quickstart

### Docker (recommended — runs everything in one command)

```bash
git clone https://github.com/Mondip-Mech/resume-agent.git
cd resume-agent

# Add your free NVIDIA NIM API key → https://build.nvidia.com
echo "NVIDIA_API_KEY=nvapi-..." > .env

docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

### Local Python

```bash
git clone https://github.com/Mondip-Mech/resume-agent.git
cd resume-agent

pip install -r requirements.txt
cp .env.example .env   # add your NVIDIA_API_KEY

streamlit run streamlit_app.py          # UI
uvicorn api.main:app --reload --port 8000  # API (separate terminal)
```

---

## REST API

The FastAPI backend is fully usable without the UI. Typical flow:

```
POST /api/jobs                       → create a session
POST /api/jobs/{id}/upload           → upload resume + JD
POST /api/jobs/{id}/run              → start pipeline (async)
GET  /api/jobs/{id}/status           → poll until complete
GET  /api/jobs/{id}/result           → scores + resume text + cover letter
GET  /api/jobs/{id}/download/resume  → download as DOCX
```

**Quick analysis endpoint** — scores a resume against a JD in < 50 ms, no LLM call needed:

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{"resume_text": "Jane Smith. Python, SQL...", "jd_text": "Senior Data Scientist..."}'
```

```json
{
  "ats_score": 61.4,
  "keyword_match": 72.0,
  "missing_keywords": ["BigQuery", "Kubernetes", "AWS"],
  "present_keywords": ["Python", "SQL", "TensorFlow"]
}
```

---

## Automated Testing

133 tests, ~15 seconds, zero API keys required (LLM calls are mocked):

```bash
pytest tests/                          # full suite
pytest tests/ --cov=. --cov-report=term-missing   # with coverage
```

| Test file | Coverage | Tests |
|---|---|---|
| `test_agents.py` | ATS scorer, data models, agents | 28 |
| `test_api.py` | All FastAPI endpoints via ASGI | 28 |
| `test_llm_client.py` | Retry logic, JSON extraction | 20 |
| `test_memory.py` | RAG, TF-IDF, feedback loop | 30 |
| `test_orchestrator.py` | Full pipeline, failure modes | 27 |

---

## CI/CD Pipeline

Every push to `main` triggers 5 automated jobs in sequence:

```
Tests (3× Python versions) → ATS Benchmark → Lint → Docker Build → Deploy to HuggingFace
```

The deploy job only runs if all four preceding jobs pass. No manual deployments.

---

## Project Structure

```
resume-agent/
├── agents/
│   ├── orchestrator.py       # Pipeline coordination + session state
│   ├── scraper_agent.py      # Resume + JD extraction
│   ├── analyzer_agent.py     # Gap analysis, ATS scoring, seniority match
│   ├── rewriter_agent.py     # Resume rewriting with RAG context
│   ├── critique_agent.py     # Self-critique loop
│   └── application_agent.py  # Cover letter + DOCX assembly
├── api/main.py               # FastAPI REST API (10 endpoints)
├── core/
│   ├── llm_client.py         # NVIDIA NIM client with smart retry
│   ├── memory.py             # ChromaDB RAG + TF-IDF fallback
│   └── models.py             # Shared Pydantic data models
├── eval/                     # ATS benchmark + pipeline evaluation CLI
├── tools/ats_scorer.py       # Heuristic ATS scorer
├── tests/                    # 133 tests
├── streamlit_app.py          # Web UI
├── Dockerfile
└── docker-compose.yml
```

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `NVIDIA_API_KEY` | Yes | Free key from [build.nvidia.com](https://build.nvidia.com) |
| `NVIDIA_MODEL` | No | Model slug (default: `meta/llama-3.1-8b-instruct`) |
| `UPLOAD_DIR` | No | Upload directory (default: `./uploads`) |
| `OUTPUT_DIR` | No | DOCX output directory (default: `./output`) |
| `CHROMA_DIR` | No | Vector store directory (default: `./chroma_db`) |

---

## License

MIT — use freely, attribution appreciated.
