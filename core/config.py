"""
core/config.py — Central configuration via Pydantic Settings.
All values are read from the .env file (or environment variables).
"""
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ── LLM — NVIDIA NIM (free tier) ───────────────────────────
    nvidia_api_key: str = Field(..., description="NVIDIA NIM API key")
    llm_model: str = "meta/llama-3.1-8b-instruct"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.2

    # ── Paths ──────────────────────────────────────────────────
    upload_dir: str = "./uploads"
    output_dir: str = "./output"
    chroma_dir: str = "./chroma_db"
    db_path: str = "./jobs.db"

    # ── Agent behavior ─────────────────────────────────────────
    max_retries: int = 3
    rewrite_iterations: int = 2
    target_ats_score: float = 80.0
    min_quantification_rate: float = 0.5

    # ── ATS ────────────────────────────────────────────────────
    top_keywords_count: int = 20
    max_resume_pages: int = 2

    # ── API ────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
