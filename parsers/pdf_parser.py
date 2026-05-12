"""
parsers/pdf_parser.py
──────────────────────
Robust PDF and DOCX text extraction for resumes and JDs.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_text(path: str | Path) -> str:
    """
    Auto-detect file type and extract clean text.
    Supports: .pdf, .docx, .doc, .txt
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return _extract_pdf(path)
    elif suffix in (".docx", ".doc"):
        return _extract_docx(path)
    elif suffix == ".txt":
        return path.read_text(encoding="utf-8", errors="replace")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _extract_pdf(path: Path) -> str:
    """Two-pass PDF extraction: PyMuPDF first, pdfplumber fallback."""
    text = ""

    # Pass 1: PyMuPDF (fast, handles most PDFs well)
    try:
        import fitz
        doc = fitz.open(str(path))
        pages = []
        for page in doc:
            pages.append(page.get_text("text", sort=True))
        doc.close()
        text = "\n".join(pages)
        if len(text.strip()) > 100:
            return _clean_text(text)
    except ImportError:
        logger.warning("PyMuPDF not installed, trying pdfplumber")
    except Exception as e:
        logger.warning(f"PyMuPDF failed: {e}")

    # Pass 2: pdfplumber (better for complex layouts)
    try:
        import pdfplumber
        with pdfplumber.open(str(path)) as pdf:
            pages = [p.extract_text() or "" for p in pdf.pages]
        text = "\n".join(pages)
        return _clean_text(text)
    except ImportError:
        raise ImportError("Install PyMuPDF or pdfplumber: pip install PyMuPDF pdfplumber")


def _extract_docx(path: Path) -> str:
    """Extract text from DOCX preserving paragraph structure."""
    try:
        from docx import Document
        doc = Document(str(path))
        paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]
        return _clean_text("\n".join(paragraphs))
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")


def _clean_text(text: str) -> str:
    """
    Remove noise while preserving structure.
    Normalises Unicode so the LLM can safely embed the text inside JSON
    strings without mis-escaping special characters.
    """
    # ── Ligatures ─────────────────────────────────────────────────────────────
    for bad, good in [("ﬁ", "fi"), ("ﬂ", "fl"), ("ﬀ", "ff"),
                      ("ﬃ", "ffi"), ("ﬄ", "ffl")]:
        text = text.replace(bad, good)

    # ── Dashes & hyphens → ASCII hyphen ───────────────────────────────────────
    for ch in ("–", "—", "‒", "―", "−"):
        # en-dash, em-dash, figure dash, horizontal bar, minus sign
        text = text.replace(ch, "-")

    # ── Quotes → ASCII ────────────────────────────────────────────────────────
    for ch in ("‘", "’", "ʼ", "`"):   # left/right single
        text = text.replace(ch, "'")
    for ch in ("“", "”", "«", "»"):   # left/right double
        text = text.replace(ch, '"')

    # ── Superscripts / subscripts → plain digits ──────────────────────────────
    _sup = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹",
                         "0123456789")
    _sub = str.maketrans("₀₁₂₃₄₅₆₇₈₉",
                         "0123456789")
    text = text.translate(_sup).translate(_sub)

    # ── Bullet / decorative symbols → plain hyphen ────────────────────────────
    for ch in ("•", "·", "●", "▪", "■",
               "◆", "★", "✦", "▸", "›", "»"):
        text = text.replace(ch, "-")

    # ── Misc characters that confuse JSON parsers ──────────────────────────────
    text = text.replace("\xa0", " ")          # non-breaking space
    text = text.replace("\x00", "")           # null bytes
    text = text.replace("�", "")         # replacement character
    text = text.replace("​", "")         # zero-width space
    text = text.replace("‌", "")         # zero-width non-joiner
    text = text.replace("‍", "")         # zero-width joiner
    text = text.replace("﻿", "")         # BOM

    # ── Whitespace normalisation ───────────────────────────────────────────────
    text = re.sub(r" {3,}", " ", text)
    text = re.sub(r"\n{4,}", "\n\n\n", text)
    # Fix hyphenated word breaks from PDF line-wrapping
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)

    return text.strip()


# ─── URL Scraping ─────────────────────────────────────────────────────────────

async def extract_from_url(url: str) -> str:
    """Scrape a job description from a URL (LinkedIn, Indeed, etc.)."""
    try:
        import httpx
        from bs4 import BeautifulSoup

        async with httpx.AsyncClient(
            headers={"User-Agent": "Mozilla/5.0 JobApplicationAgent/1.0"},
            follow_redirects=True,
            timeout=30,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()

        soup = BeautifulSoup(response.text, "lxml")

        # Remove noise
        for tag in soup.find_all(["script", "style", "nav", "footer", "header"]):
            tag.decompose()

        # Prefer article/main/job-specific containers
        main = (
            soup.find("div", {"class": re.compile(r"job.desc|description|content", re.I)})
            or soup.find("article")
            or soup.find("main")
            or soup.find("body")
        )
        text = main.get_text(separator="\n", strip=True) if main else ""
        return _clean_text(text)

    except Exception as e:
        raise RuntimeError(f"Failed to scrape URL {url}: {e}")
