"""SQLAlchemy engine/session plumbing. SQLite file by default (always
embedded — no separate server required for Tier 3 structured facts)."""
from __future__ import annotations

import os
from contextlib import contextmanager

from sqlalchemy import create_engine
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


_engine = None
_SessionLocal = None


def init_db(sqlite_path: str = "./data/tams.db"):
    global _engine, _SessionLocal
    os.makedirs(os.path.dirname(sqlite_path) or ".", exist_ok=True)
    _engine = create_engine(f"sqlite:///{sqlite_path}", connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)

    import tams.storage.models  # noqa: F401  ensure models are registered

    Base.metadata.create_all(_engine)
    return _engine


def init_db_memory():
    """In-memory SQLite — used by tests for full isolation/speed."""
    global _engine, _SessionLocal
    _engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False)
    import tams.storage.models  # noqa: F401

    Base.metadata.create_all(_engine)
    return _engine


@contextmanager
def get_session() -> Session:
    if _SessionLocal is None:
        init_db()
    session = _SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
