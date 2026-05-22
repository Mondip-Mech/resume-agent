# AI Job Application Agent

> **Multi-agent pipeline that tailors your resume to a job description, scores it against ATS criteria, self-critiques the output, and generates a personalised cover letter — powered by NVIDIA NIM (free tier).**

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://resume-agent-xqvdwdsnfnwfoq7wxhf8nt.streamlit.app/)
[![Open in HuggingFace](https://img.shields.io/badge/🤗%20HuggingFace-Space-orange)](https://huggingface.co/spaces/Mechscientist26/resume-agent)
![CI](https://github.com/Mondip-Mech/resume-agent/actions/workflows/ci.yml/badge.svg)
![Tests](https://img.shields.io/badge/tests-133%20passed-brightgreen)
![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.115-009688)
![Docker](https://img.shields.io/badge/docker-mondip007%2Fresume--agent-2496ED?logo=docker&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-lightgrey)

---

## What it does

Upload your resume and a job description. The pipeline extracts every requirement, identifies skill gaps, rewrites your resume with targeted keywords and stronger action verbs, runs an AI self-critique loop to catch weak bullets, then assembles a final application package — tailored resume + cover letter — ready to submit.

**No cloud storage. No subscription. One free NVIDIA NIM API key.**

---

## Architecture

```
                   ┌──────────────────────────────────────────┐
                   │             User Interface                │
                   │   Streamlit UI  |  FastAPI REST API       │
                   └────────┬─────────────────┬───────────────┘
                            │                 │
                            └────────┬────────┘
                                     │
                            ┌────────▼────────┐
                            │   Orchestrator   │
                            │ (session state)  │
                            └──┬──┬──┬──┬─────┘
                               │  │  │  │
          Phase 1 ─────────────┘  │  │  │
          Phase 2 ────────────────┘  │  │
          Phase 3 ───────────────────┘  │
          Phase 4 ──────────────────────┘
             │        │        │        │
      ┌──────▼──┐ ┌───▼───┐ ┌─▼──────┐ ┌──▼──────┐
      │ Scraper │ │Analyz-│ │Rewrite │ │  App    │
      │  Agent  │ │  er   │ │ Agent  │ │  Agent  │
      └─────────┘ └───────┘ └───┬────┘ └─────────┘
                                 │
                          ┌──────▼──────┐
                          │  Critique   │  <-- self-critique loop
                          │    Agent    │  (re-runs if not passed)
                          └─────────────┘
                                 │
                   ┌─────────────▼──────────────┐
                   │         Agent Memory        │
                   │  ChromaDB  |  RAG bullets   │
                   │  sentence-transformers      │
                   │  Style samples | Feedback   │
                   └────────────────────────────┘
```

### Agent roles

| Agent | Input | Output |
|---|---|---|
| **ScraperAgent** | Resume file + JD file/URL | `ExtractedResume`, `ExtractedJD` |
| **AnalyzerAgent** | Extracted resume + JD | Skill gap report, ATS score, seniority match |
| **RewriterAgent** | Analysis + style memory | ATS-optimised resume text, new score |
| **CritiqueAgent** | Rewritten resume + JD | Pass/fail verdict, fix instructions |
| **ApplicationAgent** | Final resume + analysis | Cover letter, DOCX files, submission notes |

**Self-critique loop:** if the critique does not pass, the rewriter runs a second targeted fix pass, then the critique re-runs on the corrected text — so the result you see is always the final state.

---

## Key features

- **ATS scoring** — weighted heuristic (keyword match 35%, formatting 20%, section completeness 15%, quantification 15%, action verbs 15%) validated against 15 hand-labelled resumes: **86.7% band accuracy, Spearman rho = 0.871**
- **Self-critique loop** — `CritiqueAgent` reviews every rewrite; a second fix pass runs automatically if the critique fails
- **Neural RAG memory** — `sentence-transformers/all-MiniLM-L6-v2` embeds past bullets, style samples, and feedback; falls back to TF-IDF if the model is unavailable
- **Smart retry logic** — retries on 429 rate-limit and 5xx errors; fails immediately on 400 DEGRADED model errors (no 5-minute wasted retries)
- **REST API** — full FastAPI backend, usable standalone without the Streamlit UI
- **Quick analyse endpoint** — `POST /api/analyze` scores a resume in < 50 ms, no LLM call needed
- **DOCX export** — tailored resume and cover letter as downloadable Word documents
- **133 automated tests** across all agents, memory, LLM client, API, and orchestrator

---

## Quickstart

### Option A — Docker (recommended)

```bash
# 1. Clone
git clone https://github.com/Mondip-Mech/resume-agent.git
cd resume-agent

# 2. Add your free NVIDIA NIM API key (https://build.nvidia.com)
echo "NVIDIA_API_KEY=nvapi-..." > .env

# 3. Start both services
docker-compose up --build
```

| Service | URL |
|---|---|
| Streamlit UI | http://localhost:8501 |
| FastAPI docs | http://localhost:8000/docs |

The `all-MiniLM-L6-v2` model (~80 MB) downloads once at build time and is shared between containers via a named Docker volume.

---

### Option B — Local Python

```bash
# Python 3.10+ required
git clone https://github.com/Mondip-Mech/resume-agent.git
cd resume-agent

pip install -r requirements.txt

# Optional — enables persistent ChromaDB vector store
pip install chromadb

# Set your API key
cp .env.example .env
# Edit .env → NVIDIA_API_KEY=nvapi-...

# Streamlit UI
streamlit run streamlit_app.py

# FastAPI server (separate terminal)
uvicorn api.main:app --reload --port 8000
```

---

## Environment variables

| Variable | Required | Default | Description |
|---|---|---|---|
| `NVIDIA_API_KEY` | **Yes** | — | Free key from [build.nvidia.com](https://build.nvidia.com) |
| `NVIDIA_MODEL` | No | `meta/llama-3.1-8b-instruct` | Any NIM model slug |
| `UPLOAD_DIR` | No | `./uploads` | Where uploaded files are saved |
| `OUTPUT_DIR` | No | `./output` | Where DOCX files are written |
| `CHROMA_DIR` | No | `./chroma_db` | ChromaDB persistence directory |
| `API_HOST` | No | `0.0.0.0` | FastAPI bind host |
| `API_PORT` | No | `8000` | FastAPI bind port |

---

## REST API reference

### Typical workflow

```
POST /api/jobs                          1. Create a session → session_id
POST /api/jobs/{id}/upload              2. Upload resume (+ optional JD file)
POST /api/jobs/{id}/upload-url          2b. Or provide a JD URL instead
POST /api/jobs/{id}/run                 3. Start the pipeline (background)
GET  /api/jobs/{id}/status              4. Poll until status == "complete"
GET  /api/jobs/{id}/result              5. Scores, resume text, cover letter
GET  /api/jobs/{id}/download/resume     6a. Download resume DOCX
GET  /api/jobs/{id}/download/cover_letter  6b. Download cover letter DOCX
```

### Quick analysis (no pipeline, no API key cost)

```bash
curl -X POST http://localhost:8000/api/analyze \
  -H "Content-Type: application/json" \
  -d '{
    "resume_text": "Jane Smith. Python, SQL, TensorFlow...",
    "jd_text": "Senior Data Scientist. Python required..."
  }'
```

```json
{
  "ats_score": 61.4,
  "keyword_match": 72.0,
  "quantification": 33.3,
  "action_verb_usage": 60.0,
  "formatting": 80.0,
  "section_completeness": 75.0,
  "present_keywords": ["Python", "SQL", "TensorFlow"],
  "missing_keywords": ["BigQuery", "Kubernetes", "AWS"],
  "keywords_used": ["Python", "SQL", "TensorFlow", "BigQuery", "Kubernetes", "AWS"]
}
```

### Other endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/health` | Health check + session count by status |
| `GET` | `/api/jobs` | List all sessions |
| `GET` | `/api/jobs/{id}/status` | Poll status + last 20 log lines |
| `GET` | `/api/docs` | Interactive Swagger UI |
| `GET` | `/api/redoc` | ReDoc reference |

---

## Evaluation

### ATS scorer benchmark

Validated against 15 hand-labelled (resume, JD, human_score) pairs covering diverse scenarios — perfect keyword match, wrong tech stack, ATS-hostile formatting, passive language, missing sections, and post-rewrite ideal resumes.

```
Band accuracy : 86.7%   (target >= 80%)   PASS
Spearman rho  :  0.871  (target >= 0.80)  PASS
MAE           :  7.3 pts
```

Run it yourself:

```bash
python eval/benchmark_ats.py                # summary results
python eval/benchmark_ats.py --verbose      # per-case breakdown table
python eval/benchmark_ats.py --rationale    # design justification
```

### Full pipeline evaluation

```bash
# Dry run — validates fixtures + ATS scorer, no API key needed
python eval/run_eval.py --dry-run

# Full run on sample fixtures (uses NVIDIA NIM tokens)
python eval/run_eval.py

# Compare 8B vs 70B model
python eval/run_eval.py --compare-models

# Run on your own files
python eval/run_eval.py --resume path/to/resume.txt --jd path/to/jd.txt --json
```

---

## Running tests

```bash
pip install pytest pytest-asyncio httpx

# Full suite (133 tests, ~15 seconds, no API key needed)
pytest tests/

# Individual suites
pytest tests/test_agents.py        # ATS scorer, data models, agents   (28)
pytest tests/test_api.py           # FastAPI endpoints via ASGI         (28)
pytest tests/test_llm_client.py    # Retry logic, JSON extraction       (20)
pytest tests/test_memory.py        # RAG, TF-IDF, feedback loop         (30)
pytest tests/test_orchestrator.py  # Full pipeline, failure modes       (27)

# With coverage report
pytest tests/ --cov=. --cov-report=term-missing
```

All tests run **in-process** with mocked LLM calls — no NVIDIA API key needed, no network access required.

---

## Project structure

```
job-agent/
├── agents/
│   ├── orchestrator.py       # Coordinates all phases + session state
│   ├── scraper_agent.py      # JD + resume extraction via LLM
│   ├── analyzer_agent.py     # Gap analysis, seniority match, ATS score
│   ├── rewriter_agent.py     # Resume rewriting with RAG context
│   ├── critique_agent.py     # Self-critique loop (pass/fail + fix)
│   └── application_agent.py  # Cover letter generation + DOCX assembly
├── api/
│   └── main.py               # FastAPI REST API (10 endpoints, v2.1)
├── core/
│   ├── config.py             # Settings via pydantic-settings + .env
│   ├── llm_client.py         # NVIDIA NIM client, retry, JSON repair
│   ├── memory.py             # ChromaDB RAG, TF-IDF fallback, feedback
│   └── models.py             # All Pydantic data models (shared)
├── eval/
│   ├── benchmark_ats.py      # 15-case ATS scorer benchmark + Spearman
│   ├── evaluator.py          # PipelineEvaluator class
│   ├── metrics.py            # 30-field metric definitions + aggregation
│   ├── run_eval.py           # CLI: --dry-run, --compare-models, --json
│   └── fixtures/             # sample_resume.txt + sample_jd.txt
├── parsers/
│   └── pdf_parser.py         # PDF/DOCX/URL text extraction
├── tools/
│   └── ats_scorer.py         # Heuristic ATS scorer (keyword + format)
├── tests/                    # 133 tests — pytest + pytest-asyncio + httpx
├── streamlit_app.py          # Streamlit UI (non-blocking async pipeline)
├── Dockerfile
├── docker-compose.yml        # API + Streamlit + shared HF cache volume
└── requirements.txt
```

---

## How the ATS score is calculated

```
ATS score  =  keyword_match       x 0.35
           +  formatting          x 0.20
           +  section_complete    x 0.15
           +  quantification      x 0.15
           +  action_verb_usage   x 0.15
```

| Component | What it checks |
|---|---|
| **keyword_match** | Fraction of JD keywords present in resume (TF-IDF extracted, stop-words filtered) |
| **formatting** | Penalises tables, text boxes, excessive special characters that confuse ATS parsers |
| **section_completeness** | Presence of Summary, Experience, Education, Skills sections |
| **quantification** | Ratio of bullets containing numbers, percentages, or currency values |
| **action_verb_usage** | Ratio of experience bullets starting with a recognised action verb |

Target score for pipeline to mark "ready to submit": **>= 75 / 100**

---

## Tech stack

| Layer | Technology |
|---|---|
| LLM | NVIDIA NIM — `meta/llama-3.1-8b-instruct` (free tier) |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (local, ~80 MB) |
| Vector store | ChromaDB (persistent) with TF-IDF in-memory fallback |
| API | FastAPI + Uvicorn |
| UI | Streamlit |
| Data models | Pydantic v2 with field validators |
| Retry logic | Tenacity (smart — skips retrying DEGRADED 400 errors) |
| Document export | python-docx |
| PDF parsing | PyMuPDF -> pdfplumber fallback |
| Testing | pytest + pytest-asyncio + httpx ASGITransport |
| Containers | Docker + docker-compose |

---

## CI/CD

GitHub Actions ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)) runs on every push to `main`:

| Job | What it does |
|---|---|
| **Tests** | Full 133-test suite across Python 3.10 / 3.11 / 3.12 with mocked LLM calls |
| **Benchmark** | ATS scorer validated against 15 hand-labelled resumes (band accuracy ≥ 80%, Spearman rho ≥ 0.80) |
| **Lint** | `ruff check` (pycodestyle, pyflakes, isort) |
| **Docker build** | Builds the image with BuildKit cache — no push, verifies it compiles |
| **Deploy** | Pushes to [HuggingFace Space](https://huggingface.co/spaces/Mechscientist26/resume-agent) only after all above jobs pass |

**Required GitHub secret:** add `HF_TOKEN` (a HuggingFace write token) in **Settings → Secrets and variables → Actions** in the GitHub repo.

---

## License

MIT — use freely, attribution appreciated.
