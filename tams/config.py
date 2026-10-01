"""Central configuration for TAMS v2, loaded from environment / .env."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="TAMS_", extra="ignore")

    # --- LLM ---
    openai_api_key: str | None = None
    orchestrator_model: str = "gpt-4.1"
    verifier_model: str = "gpt-4.1"
    extractor_model: str = "gpt-4.1-mini"
    reranker_model: str = "gpt-4.1-mini"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = 1536

    # --- Infra ---
    redis_url: str = "redis://localhost:6379/0"
    qdrant_url: str = "http://localhost:6333"
    sqlite_path: str = "./data/tams.db"

    # --- Tier 1 ---
    context_window_tokens: int = 128_000
    working_memory_budget_fraction: float = 0.65
    working_memory_max_turns: int = 200

    # --- Tier 2 ---
    chunk_size_turns: int = 3

    # --- Tier 3 ---
    preference_batch_interval_turns: int = 20
    extractor_circuit_breaker_threshold: int = 5
    extractor_circuit_breaker_cooldown_s: float = 60.0

    # --- Retrieval ---
    retrieval_pool_size: int = 2000
    retrieval_top_k: int = 12

    # --- Decay ---
    decay_half_life_turns: float = 200.0

    # --- Verifier ---
    hallucination_regen_threshold: float = 0.5  # trigger one bounded regen pass above this rate

    # --- Cost ---
    cost_input_per_1k: float = 0.0
    cost_output_per_1k: float = 0.0


def load_settings() -> Settings:
    """Load settings. `openai_api_key` intentionally also accepts the
    unprefixed standard `OPENAI_API_KEY` env var (most tooling expects that
    name), in addition to `TAMS_OPENAI_API_KEY`."""
    import os

    base = Settings()
    key = os.environ.get("OPENAI_API_KEY")
    if key:
        base.openai_api_key = key
    return base


settings = load_settings()
