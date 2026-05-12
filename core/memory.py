"""
core/memory.py
───────────────
Multi-purpose vector + key-value memory for the agent system.

Stores:
  • Resume bullet points (bullet-level RAG — highest impact)
  • Job description chunks
  • Interview feedback history
  • User style profile (best bullets / tone)
  • Job market intelligence (consensus keywords from multiple JDs)

ChromaDB is used when available. Falls back to an in-memory TF-IDF
similarity store so RAG works even without ChromaDB installed.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

logger = logging.getLogger(__name__)


# ─── Neural embedding function (sentence-transformers) ───────────────────────

def _get_embedding_function():
    """
    Return a ChromaDB-compatible embedding function backed by sentence-transformers
    (all-MiniLM-L6-v2, 80 MB, runs locally, no API key needed).

    Falls back to None if sentence-transformers is not installed, in which case
    ChromaDB uses its own default embeddings, and the TF-IDF fallback uses
    keyword overlap instead of neural similarity.
    """
    try:
        from chromadb.utils.embedding_functions import (
            SentenceTransformerEmbeddingFunction,
        )
        fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        logger.info("Neural embeddings: sentence-transformers via ChromaDB utility.")
        return fn
    except Exception:
        pass

    try:
        from sentence_transformers import SentenceTransformer

        class _STEmbedFn:
            """Minimal ChromaDB-compatible wrapper around SentenceTransformer."""
            def __init__(self):
                self._model = SentenceTransformer("all-MiniLM-L6-v2")

            def __call__(self, input: list) -> list:  # noqa: A002
                vecs = self._model.encode(input, convert_to_numpy=True)
                return vecs.tolist()

        logger.info("Neural embeddings: sentence-transformers (direct).")
        return _STEmbedFn()
    except ImportError:
        logger.info(
            "sentence-transformers not installed — using TF-IDF similarity. "
            "For better RAG quality: pip install sentence-transformers"
        )
        return None


# ─── Simple TF-IDF fallback (no extra deps) ──────────────────────────────────

class _TFIDFStore:
    """
    Lightweight in-memory vector store using TF-IDF + cosine similarity.
    Used when ChromaDB is unavailable.  No neural embeddings needed.
    """
    def __init__(self):
        self._docs: List[str] = []
        self._metas: List[dict] = []
        self._ids: List[str] = []

    def upsert(self, documents, ids, metadatas=None):
        metas = metadatas or [{} for _ in documents]
        for doc, did, meta in zip(documents, ids, metas):
            if did in self._ids:
                idx = self._ids.index(did)
                self._docs[idx] = doc
                self._metas[idx] = meta
            else:
                self._docs.append(doc)
                self._ids.append(did)
                self._metas.append(meta)

    def query(self, query_texts, n_results=5, embedding_fn=None):
        if not self._docs:
            return {"documents": [[]], "metadatas": [[]]}

        if embedding_fn is not None:
            # ── Neural cosine similarity ───────────────────────
            try:
                import numpy as np
                q_vec  = np.array(embedding_fn([query_texts[0]])[0])
                d_vecs = np.array(embedding_fn(self._docs))
                q_norm = q_vec  / (np.linalg.norm(q_vec)  + 1e-10)
                d_norm = d_vecs / (np.linalg.norm(d_vecs, axis=1, keepdims=True) + 1e-10)
                scores = (d_norm @ q_norm).tolist()
            except Exception:
                # Fall through to TF-IDF on any error
                embedding_fn = None

        if embedding_fn is None:
            # ── TF-IDF cosine similarity (keyword overlap) ─────
            q_tokens = self._tokenize(query_texts[0].lower())
            scores   = [self._cosine(q_tokens, self._tokenize(d.lower())) for d in self._docs]

        top = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:n_results]
        return {
            "documents": [[self._docs[i] for i in top]],
            "metadatas": [[self._metas[i] for i in top]],
        }

    def count(self):
        return len(self._docs)

    @staticmethod
    def _tokenize(text: str) -> Counter:
        words = re.findall(r"\b\w+\b", text.lower())
        return Counter(words)

    @staticmethod
    def _cosine(a: Counter, b: Counter) -> float:
        if not a or not b:
            return 0.0
        dot = sum(a[k] * b.get(k, 0) for k in a)
        mag_a = math.sqrt(sum(v * v for v in a.values()))
        mag_b = math.sqrt(sum(v * v for v in b.values()))
        return dot / (mag_a * mag_b) if mag_a and mag_b else 0.0


class _TFIDFCollection:
    def __init__(self): self._store = _TFIDFStore()
    def upsert(self, **kwargs): self._store.upsert(**kwargs)
    def query(self, embedding_fn=None, **kwargs):
        return self._store.query(embedding_fn=embedding_fn, **kwargs)
    def count(self): return self._store.count()


class _InMemoryClient:
    """In-memory client mirroring ChromaDB API — used as fallback."""
    def __init__(self):
        self._cols: Dict[str, _TFIDFCollection] = {}

    def get_or_create_collection(self, name, **kwargs):
        if name not in self._cols:
            self._cols[name] = _TFIDFCollection()
        return self._cols[name]

    def delete_collection(self, name):
        self._cols.pop(name, None)


# ─── Main AgentMemory ─────────────────────────────────────────────────────────

class AgentMemory:
    """
    Per-session vector store with bullet-level RAG, feedback history,
    style profile, and job market intelligence.

    Collection naming:
      {session_id}_jd        — job description chunks
      {session_id}_resume    — resume chunks
      {session_id}_bullets   — individual resume bullets (RAG)
      global_feedback        — interview outcome history
      global_style           — user's best bullets / style
      global_market          — job market keyword intelligence
    """

    def __init__(self, persist_dir: str = "./chroma_db"):
        self._persist_dir = persist_dir
        self._client = None
        self._embedding_fn = _get_embedding_function()   # neural or None
        # Persistent stores that survive between sessions
        self._feedback_store: List[dict] = []
        self._style_bullets: List[str] = []
        self._market_jobs: List[str] = []
        self._load_persistent()

    # ─── Client init ──────────────────────────────────────────

    def _get_client(self):
        if self._client is None:
            try:
                import chromadb
                self._client = chromadb.PersistentClient(path=self._persist_dir)
                logger.info("ChromaDB initialized (persistent).")
            except ImportError:
                logger.warning("ChromaDB not installed — using in-memory TF-IDF store.")
                self._client = _InMemoryClient()
            except Exception as e:
                logger.warning(f"ChromaDB init failed ({e}) — using in-memory store.")
                self._client = _InMemoryClient()
        return self._client

    def _col(self, name: str):
        kwargs: dict = {"name": name, "metadata": {"hnsw:space": "cosine"}}
        if self._embedding_fn is not None:
            kwargs["embedding_function"] = self._embedding_fn
        return self._get_client().get_or_create_collection(**kwargs)

    @staticmethod
    def _safe_name(session_id: str, suffix: str) -> str:
        return re.sub(r"[^a-zA-Z0-9_-]", "", session_id)[:32] + f"_{suffix}"

    # ─── Session-level storage ────────────────────────────────

    def store_jd(self, session_id: str, text: str):
        self._store_chunks(self._safe_name(session_id, "jd"), text)

    def store_resume(self, session_id: str, text: str):
        self._store_chunks(self._safe_name(session_id, "resume"), text)

    def search_jd(self, session_id: str, query: str, top_k: int = 5) -> List[str]:
        return self._search(self._safe_name(session_id, "jd"), query, top_k)

    def search_resume(self, session_id: str, query: str, top_k: int = 5) -> List[str]:
        return self._search(self._safe_name(session_id, "resume"), query, top_k)

    # ─── Bullet-level RAG ─────────────────────────────────────

    def store_bullets(self, session_id: str, bullets: List[str]):
        """
        Store individual resume bullet points for fine-grained RAG retrieval.
        Also adds them to the global style store so future sessions benefit.
        """
        if not bullets:
            return
        col_name = self._safe_name(session_id, "bullets")
        col = self._col(col_name)
        ids, docs, metas = [], [], []
        for i, bullet in enumerate(bullets):
            b = bullet.strip()
            if not b:
                continue
            did = hashlib.md5(f"{session_id}_b_{i}_{b[:40]}".encode()).hexdigest()
            ids.append(did)
            docs.append(b)
            metas.append({"session_id": session_id, "bullet_index": i})
        if ids:
            try:
                col.upsert(documents=docs, ids=ids, metadatas=metas)
            except Exception as e:
                logger.warning(f"Bullet store failed: {e}")

        # Also add to global style store
        self._style_bullets.extend(bullets)
        self._save_persistent()

    def retrieve_relevant_bullets(self, session_id: str, requirement: str, top_k: int = 4) -> List[str]:
        """
        Retrieve the most semantically similar bullet points from the resume
        for a given JD requirement. Used to give the rewriter concrete evidence.
        """
        col_name = self._safe_name(session_id, "bullets")
        return self._search(col_name, requirement, top_k)

    def extract_and_store_bullets(self, session_id: str, resume_text: str):
        """
        Parse bullet points from raw resume text and store them.
        Called automatically during the scraping phase.
        """
        bullets = []
        for line in resume_text.split("\n"):
            line = line.strip()
            # Lines that start with bullet markers or are experience bullets
            if line.startswith(("-", "•", "*", "–")) and len(line) > 15:
                bullets.append(line.lstrip("-•*– ").strip())
            # Lines that start with action verbs (experience bullets without markers)
            elif len(line) > 20 and line[0].isupper() and not line.endswith(":"):
                words = line.split()
                if words and words[0].lower() in {
                    "built","developed","designed","led","managed","created",
                    "implemented","deployed","optimized","reduced","increased",
                    "achieved","delivered","analysed","analyzed","automated",
                    "collaborated","improved","launched","scaled","streamlined",
                    "engineered","architected","migrated","trained","mentored",
                }:
                    bullets.append(line)
        self.store_bullets(session_id, bullets)
        logger.info(f"  RAG: stored {len(bullets)} bullets for session {session_id}")
        return bullets

    # ─── Interview Feedback Loop ───────────────────────────────

    def store_feedback(self, session_id: str, job_title: str, company: str,
                       outcome: str, notes: str = ""):
        """
        Store the outcome of a job application for future reference.
        Outcomes: 'got_interview', 'got_offer', 'rejected', 'no_response'
        """
        entry = {
            "session_id": session_id,
            "job_title": job_title,
            "company": company,
            "outcome": outcome,
            "notes": notes,
        }
        self._feedback_store.append(entry)
        self._save_persistent()
        logger.info(f"Feedback stored: {job_title} at {company} → {outcome}")

    def get_feedback_insights(self, similar_role: str = "") -> str:
        """
        Return a human-readable summary of past application outcomes.
        Used to inform the analyzer and rewriter about what worked before.
        """
        if not self._feedback_store:
            return ""

        outcomes = Counter(e["outcome"] for e in self._feedback_store)
        total = len(self._feedback_store)
        interview_rate = outcomes.get("got_interview", 0) + outcomes.get("got_offer", 0)

        lines = [
            f"Past applications: {total} total, "
            f"{interview_rate} interviews ({100*interview_rate//total}% rate).",
        ]

        # Find relevant past notes
        if similar_role:
            relevant = [
                e for e in self._feedback_store
                if similar_role.lower() in e.get("job_title", "").lower()
                and e.get("notes")
            ]
            for e in relevant[-3:]:
                lines.append(
                    f"- {e['job_title']} at {e['company']} → {e['outcome']}: {e['notes']}"
                )

        return "\n".join(lines)

    def get_all_feedback(self) -> List[dict]:
        return list(self._feedback_store)

    # ─── Personal Style Learning ───────────────────────────────

    def store_style_samples(self, bullets: List[str]):
        """Store the user's own best bullet points to learn their writing style."""
        self._style_bullets = list(set(self._style_bullets + bullets))
        self._save_persistent()

    def get_style_context(self, top_k: int = 6) -> str:
        """
        Return sample bullets in the user's own voice to guide the rewriter
        toward matching their authentic style.
        """
        if not self._style_bullets:
            return ""
        samples = self._style_bullets[-top_k:]
        return "USER'S WRITING STYLE (match this tone and structure):\n" + \
               "\n".join(f"- {b}" for b in samples)

    def analyze_style(self) -> dict:
        """Extract style statistics from stored bullets."""
        if not self._style_bullets:
            return {}
        word_counts = [len(b.split()) for b in self._style_bullets]
        first_words = [b.split()[0].lower() for b in self._style_bullets if b.split()]
        has_numbers = sum(1 for b in self._style_bullets if re.search(r"\d", b))
        return {
            "avg_bullet_length": round(sum(word_counts) / len(word_counts)),
            "uses_metrics": has_numbers / len(self._style_bullets) > 0.3,
            "top_verbs": [w for w, _ in Counter(first_words).most_common(5)],
            "total_samples": len(self._style_bullets),
        }

    # ─── Job Market Intelligence ───────────────────────────────

    def store_job_posting(self, jd_text: str):
        """
        Store a job posting for market analysis.
        Call multiple times with different JDs for the same target role.
        """
        self._market_jobs.append(jd_text[:3000])
        self._save_persistent()

    def get_market_keywords(self, top_k: int = 20) -> List[str]:
        """
        Return keywords that appear most frequently across stored job postings.
        These are the 'consensus' skills the market demands for this role.
        """
        if not self._market_jobs:
            return []
        word_counter: Counter = Counter()
        for jd in self._market_jobs:
            words = re.findall(r"\b[A-Za-z][A-Za-z+#.]{2,20}\b", jd)
            word_counter.update(w.lower() for w in words)

        # Filter out stop words
        stop = {"the","and","for","with","this","that","have","from","your","will",
                "you","our","they","what","who","how","all","can","not","but",
                "has","been","one","any","may","also","more","than","such",
                "experience","required","preferred","skills","ability","role"}
        return [w for w, _ in word_counter.most_common(top_k * 2)
                if w not in stop and len(w) > 2][:top_k]

    def get_market_summary(self) -> str:
        """Human-readable summary of job market intelligence."""
        if not self._market_jobs:
            return ""
        kw = self.get_market_keywords(15)
        return (
            f"Market Intelligence ({len(self._market_jobs)} job postings analysed):\n"
            f"Top consensus skills: {', '.join(kw)}"
        )

    # ─── Persistence ──────────────────────────────────────────

    def _get_persist_path(self) -> Path:
        p = Path(self._persist_dir)
        p.mkdir(parents=True, exist_ok=True)
        return p / "agent_memory.json"

    def _save_persistent(self):
        try:
            data = {
                "feedback": self._feedback_store,
                "style_bullets": self._style_bullets[-100:],   # keep last 100
                "market_jobs": self._market_jobs[-50:],         # keep last 50
            }
            self._get_persist_path().write_text(
                json.dumps(data, indent=2), encoding="utf-8"
            )
        except Exception as e:
            logger.warning(f"Memory persist failed: {e}")

    def _load_persistent(self):
        try:
            path = self._get_persist_path()
            if path.exists():
                data = json.loads(path.read_text(encoding="utf-8"))
                self._feedback_store = data.get("feedback", [])
                self._style_bullets  = data.get("style_bullets", [])
                self._market_jobs    = data.get("market_jobs", [])
                logger.info(
                    f"Memory loaded: {len(self._feedback_store)} feedback entries, "
                    f"{len(self._style_bullets)} style bullets."
                )
        except Exception:
            pass  # Fresh start is fine

    # ─── Session cleanup ──────────────────────────────────────

    def clear_session(self, session_id: str):
        client = self._get_client()
        for suffix in ("jd", "resume", "bullets"):
            try:
                client.delete_collection(self._safe_name(session_id, suffix))
            except Exception:
                pass

    # ─── Internal helpers ─────────────────────────────────────

    def _store_chunks(self, col_name: str, text: str, chunk_size: int = 300):
        words = text.split()
        if not words:
            return
        chunks, step = [], max(1, chunk_size - 50)
        for i in range(0, len(words), step):
            c = " ".join(words[i: i + chunk_size])
            if c.strip():
                chunks.append(c)
        if not chunks:
            return
        col = self._col(col_name)
        ids = [hashlib.md5(f"{col_name}_{i}".encode()).hexdigest() for i in range(len(chunks))]
        try:
            col.upsert(
                documents=chunks, ids=ids,
                metadatas=[{"chunk": i} for i in range(len(chunks))],
            )
        except Exception as e:
            logger.warning(f"Chunk store failed for {col_name}: {e}")

    def _search(self, col_name: str, query: str, top_k: int) -> List[str]:
        try:
            col = self._col(col_name)
            n   = min(top_k, max(1, col.count()))
            # Only pass embedding_fn to the in-memory TF-IDF store.
            # ChromaDB's real Collection.query() does not accept this kwarg.
            query_kwargs: dict = {"query_texts": [query], "n_results": n}
            if isinstance(self._get_client(), _InMemoryClient):
                query_kwargs["embedding_fn"] = self._embedding_fn
            results = col.query(**query_kwargs)
            return results.get("documents", [[]])[0]
        except Exception as e:
            logger.warning(f"Memory search failed for {col_name}: {e}")
            return []
