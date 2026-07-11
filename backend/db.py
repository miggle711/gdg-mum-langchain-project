import logging
import os
from typing import Optional

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.orm import declarative_base

from app.config import settings

logger = logging.getLogger(__name__)

DATABASE_URL = settings.database_url

Base = declarative_base()

_engine: Optional[AsyncEngine] = None
_SessionLocal: Optional[async_sessionmaker] = None


def get_engine() -> AsyncEngine:
    global _engine
    if _engine is None:
        _engine = create_async_engine(DATABASE_URL, pool_pre_ping=True)
    return _engine


def get_session() -> AsyncSession:
    """Returns a new AsyncSession bound to the lazily-created engine.
    Caller is responsible for closing it (use `async with get_session() as session:`)."""
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = async_sessionmaker(bind=get_engine(), expire_on_commit=False)
    return _SessionLocal()


async def init_db() -> None:
    """Idempotent table creation for local dev/test convenience.
    Base.metadata.create_all() only creates tables that don't already
    exist — it never alters existing ones, so it's not a substitute for
    Alembic migrations (see run_migrations() below). Used directly by
    SQLite-backed unit tests that don't want to run real migrations."""
    engine = get_engine()
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


def run_migrations() -> None:
    """Applies any pending Alembic migrations. Safe to call on every boot —
    a no-op if the schema is already at head.

    This function itself stays sync: Alembic's migration runner drives its
    own internal event loop (see alembic/env.py's run_migrations_online()),
    so it can't be awaited from within an already-running event loop. Call
    this from app/main.py before the ASGI app starts serving requests, not
    from inside a route handler.
    """
    from alembic.config import Config
    from alembic import command

    alembic_cfg = Config(os.path.join(os.path.dirname(__file__), "alembic.ini"))
    alembic_cfg.set_main_option("sqlalchemy.url", DATABASE_URL)
    command.upgrade(alembic_cfg, "head")
