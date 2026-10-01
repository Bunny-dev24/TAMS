import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("TAMS_SQLITE_PATH", "./data/test_tams.db")

from tams.config import Settings  # noqa: E402
from tams.llm.circuit_breaker import reset_all_breakers  # noqa: E402
from tams.session import reset_shared_state  # noqa: E402
from tams.storage import db as db_module  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_state():
    reset_shared_state()
    reset_all_breakers()
    db_module.init_db_memory()
    yield
    reset_shared_state()
    reset_all_breakers()


@pytest.fixture
def settings() -> Settings:
    s = Settings(
        openai_api_key=None,
        working_memory_budget_fraction=0.65,
        context_window_tokens=2000,  # small, so compaction triggers quickly in tests
        working_memory_max_turns=20,
        chunk_size_turns=3,
        preference_batch_interval_turns=4,
        retrieval_pool_size=50,
        retrieval_top_k=5,
        decay_half_life_turns=50,
    )
    return s
