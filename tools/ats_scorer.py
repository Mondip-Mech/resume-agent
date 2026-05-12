"""
tools/ats_scorer.py
────────────────────
Heuristic ATS (Applicant Tracking System) compatibility scorer.

Scores 0–100 across five dimensions:
  1. Keyword match    — % of JD keywords present in resume
  2. Formatting       — structure, section presence, no tables/columns
  3. Section complete — has all required sections
  4. Quantification   — % of bullets with numbers
  5. Action verbs     — % of bullets starting with strong verbs
"""
from __future__ import annotations

import re
from typing import List, Set

from core.models import ATSScore

# Strong action verbs recognized by ATS
_ACTION_VERBS = {
    "achieved", "administered", "analyzed", "architected", "automated",
    "built", "championed", "collaborated", "consolidated", "created",
    "delivered", "deployed", "designed", "developed", "directed",
    "drove", "engineered", "established", "executed", "generated",
    "grew", "implemented", "improved", "increased", "launched",
    "led", "managed", "mentored", "migrated", "optimized",
    "orchestrated", "owned", "partnered", "piloted", "pioneered",
    "produced", "reduced", "refactored", "resolved", "scaled",
    "shipped", "simplified", "spearheaded", "streamlined", "transformed",
}

# Sections an ATS expects
_EXPECTED_SECTIONS = {
    "experience", "education", "skills", "summary", "objective",
    "work", "employment", "projects", "certifications", "achievements",
}

# Patterns that confuse ATS (tables, columns)
_ATS_HOSTILE_PATTERNS = [
    re.compile(r"\|.*\|"),          # Pipe-separated tables
    re.compile(r"\t.*\t"),          # Tab-separated columns
]


def score_resume(resume_text: str, jd_keywords: List[str]) -> ATSScore:
    """
    Compute ATS compatibility score for a resume against a list of JD keywords.

    Parameters
    ----------
    resume_text : str
        Full text of the resume (after any rewriting).
    jd_keywords : List[str]
        Keywords extracted from the job description.

    Returns
    -------
    ATSScore with overall 0-100 and sub-scores.
    """
    lower_text = resume_text.lower()
    lines = [ln.strip() for ln in resume_text.split("\n") if ln.strip()]
    bullets = [ln for ln in lines if ln.startswith(("•", "-", "*", "–")) or
               (len(ln) > 20 and not ln.endswith(":"))]

    # 1. Keyword match
    present = [kw for kw in jd_keywords if kw.lower() in lower_text]
    missing = [kw for kw in jd_keywords if kw.lower() not in lower_text]
    kw_score = (len(present) / len(jd_keywords) * 100) if jd_keywords else 50.0

    # 2. Formatting score
    formatting_score = _score_formatting(resume_text, lines)

    # 3. Section completeness
    sections_found = {
        sec for sec in _EXPECTED_SECTIONS
        if re.search(rf"\b{sec}\b", lower_text)
    }
    section_score = min(100.0, len(sections_found) / 4 * 100)

    # 4. Quantification rate
    bullets_with_numbers = sum(
        1 for b in bullets if re.search(r"\d+[%$k+x]?|\$\d+|\d+\s*(users|clients|team)", b, re.I)
    )
    quant_score = (bullets_with_numbers / len(bullets) * 100) if bullets else 0.0

    # 5. Action verb usage
    bullets_with_verbs = sum(
        1 for b in bullets
        if _starts_with_action_verb(b)
    )
    verb_score = (bullets_with_verbs / len(bullets) * 100) if bullets else 0.0

    # Weighted overall
    overall = (
        kw_score       * 0.35 +
        formatting_score * 0.20 +
        section_score  * 0.15 +
        quant_score    * 0.15 +
        verb_score     * 0.15
    )

    return ATSScore(
        overall=round(overall, 1),
        keyword_match=round(kw_score, 1),
        formatting=round(formatting_score, 1),
        section_completeness=round(section_score, 1),
        quantification=round(quant_score, 1),
        action_verb_usage=round(verb_score, 1),
        missing_keywords=missing,
        present_keywords=present,
    )


def _score_formatting(text: str, lines: List[str]) -> float:
    score = 100.0

    # Penalize ATS-hostile patterns
    for pattern in _ATS_HOSTILE_PATTERNS:
        if pattern.search(text):
            score -= 20

    # Penalize if very few line breaks (likely all one block)
    if len(lines) < 10:
        score -= 15

    # Penalize excessive special characters (likely decorative)
    special_count = len(re.findall(r"[★✦◆▪■●]", text))
    if special_count > 10:
        score -= 10

    return max(0.0, score)


def _starts_with_action_verb(bullet: str) -> bool:
    """Check if a bullet point starts with a recognized action verb."""
    # Strip bullet markers
    clean = re.sub(r"^[•\-\*–]\s*", "", bullet).strip()
    first_word = clean.split()[0].lower().rstrip(".,;") if clean else ""
    return first_word in _ACTION_VERBS


def extract_jd_keywords(jd_text: str, top_k: int = 25) -> List[str]:
    """
    Simple keyword extraction from JD text using frequency + position weighting.
    Falls back to frequency if spaCy is unavailable.
    """
    try:
        return _spacy_keywords(jd_text, top_k)
    except Exception:
        return _regex_keywords(jd_text, top_k)


def _spacy_keywords(text: str, top_k: int) -> List[str]:
    """NLP-based noun phrase extraction."""
    import spacy
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        raise RuntimeError("spaCy model not found")

    doc = nlp(text[:10000])
    # Noun chunks + named entities
    phrases: Set[str] = set()
    for chunk in doc.noun_chunks:
        if 2 <= len(chunk.text.split()) <= 4:
            phrases.add(chunk.text.lower())
    for ent in doc.ents:
        phrases.add(ent.text.lower())

    # Also add single important tokens (skills, tools)
    for token in doc:
        if token.pos_ in ("PROPN", "NOUN") and not token.is_stop and len(token.text) > 2:
            phrases.add(token.lemma_.lower())

    # Sort by position in text (earlier = more important)
    scored = sorted(phrases, key=lambda p: text.lower().find(p))
    return scored[:top_k]


def _regex_keywords(text: str, top_k: int) -> List[str]:
    """
    Fallback keyword extractor: pulls real skills and technologies from JD text.
    Filters out common English stop-words so junk words like 'What', 'They',
    'About', 'You' are never returned as keywords.
    """

    # Common English words that are NOT skills — always exclude these
    _STOP_WORDS = {
        "the", "and", "for", "are", "with", "this", "that", "have", "from",
        "your", "will", "you", "our", "they", "their", "what", "who", "how",
        "all", "can", "its", "not", "but", "was", "been", "has", "had",
        "one", "any", "may", "also", "more", "than", "such", "both",
        "each", "into", "about", "when", "where", "there", "here",
        "which", "would", "could", "should", "other", "some", "well",
        "new", "use", "work", "team", "role", "job", "company", "inc",
        "ltd", "llc", "corp", "pvt", "candidates", "candidate", "position",
        "experience", "required", "preferred", "skills", "ability",
        "knowledge", "understanding", "strong", "good", "great", "excellent",
        "looking", "join", "help", "need", "want", "make", "using",
        "including", "ensure", "provide", "across", "within", "between",
        # Single-letter and very short non-technical words
        "ii", "iii", "iv",
    }

    # Job seniority / role title words — phrases made entirely of these
    # are job titles (e.g. "Senior Backend Engineer"), not resume keywords
    _TITLE_WORDS = {
        "senior", "junior", "lead", "staff", "principal", "associate",
        "engineer", "developer", "manager", "director", "analyst",
        "scientist", "architect", "consultant", "specialist", "coordinator",
        "officer", "head", "vp", "president", "intern", "contractor",
        "backend", "frontend", "fullstack", "full", "stack", "software",
        "platform", "infrastructure", "site", "reliability", "data",
    }

    # Explicit tech/skill keyword patterns — always include if present
    tech_pattern = re.compile(
        r"\b(?:Python|Java|JavaScript|TypeScript|SQL|NoSQL|R\b|Scala|Go|"
        r"AWS|GCP|Azure|S3|EC2|Lambda|BigQuery|Redshift|Snowflake|"
        r"React|Node|Vue|Angular|Django|Flask|FastAPI|Spring|"
        r"Docker|Kubernetes|Terraform|Airflow|Spark|Hadoop|Kafka|"
        r"TensorFlow|PyTorch|Keras|scikit-learn|XGBoost|LightGBM|"
        r"Tableau|PowerBI|Power\s*BI|Looker|Metabase|Grafana|"
        r"PostgreSQL|MySQL|MongoDB|Redis|Elasticsearch|DynamoDB|"
        r"ML|AI|NLP|CV|LLM|RAG|MLOps|DevOps|"
        r"CI/CD|REST|API|GraphQL|gRPC|"
        r"Agile|Scrum|Kanban|JIRA|Confluence|"
        r"ETL|ELT|dbt|Fivetran|Airbyte|"
        r"SaaS|B2B|B2C|FinTech|EdTech|"
        r"Excel|Sheets|Pandas|NumPy|Matplotlib|Seaborn|"
        r"Git|GitHub|GitLab|Bitbucket|"
        r"Hadoop|Hive|Pig|HBase|Cassandra)\b",
        re.IGNORECASE,
    )

    tech_matches = tech_pattern.findall(text)

    # Also pick up capitalized noun phrases (2-3 words) that look like skill names
    # e.g. "Machine Learning", "Data Science", "A/B Testing"
    phrases = re.findall(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,2}\b", text)
    phrase_cleaned = [
        p for p in phrases
        if p.lower() not in _STOP_WORDS
        and not any(w.lower() in _STOP_WORDS for w in p.split())
        and len(p) > 4
        # Skip phrases made entirely of job-title words (e.g. "Senior Backend Engineer")
        and not all(w.lower() in _TITLE_WORDS for w in p.split())
    ]

    all_kw = [t.strip() for t in tech_matches] + phrase_cleaned

    # Deduplicate preserving original casing; strip stray backticks and whitespace
    seen = set()
    result = []
    for kw in all_kw:
        kw = kw.strip().strip("`").strip()   # remove any accidental backticks
        if not kw:
            continue
        key = kw.lower()
        if key not in seen and key not in _STOP_WORDS and len(key) > 2:
            seen.add(key)
            result.append(kw)
        if len(result) >= top_k:
            break

    return result
