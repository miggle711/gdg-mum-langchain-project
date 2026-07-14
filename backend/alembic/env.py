import asyncio
import os
import sys
from logging.config import fileConfig

from sqlalchemy import pool
from sqlalchemy.ext.asyncio import async_engine_from_config

from alembic import context

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # backend/ on path, same boilerplate as app/main.py

from db import Base
import models_db  # noqa: F401 — import side effect registers all model classes on Base.metadata
from app.config import settings

# this is the Alembic Config object, which provides
# access to the values within the .ini file in use.
config = context.config
config.set_main_option("sqlalchemy.url", settings.database_url)

# Interpret the config file for Python logging.
# This line sets up loggers basically.
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata

# other values from the config, defined by the needs of env.py,
# can be acquired:
# my_important_option = config.get_main_option("my_important_option")
# ... etc.


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    This configures the context with just a URL
    and not an Engine, though an Engine is acceptable
    here as well.  By skipping the Engine creation
    we don't even need a DBAPI to be available.

    Calls to context.execute() here emit the given string to the
    script output.

    """
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
    )

    with context.begin_transaction():
        context.run_migrations()


def do_run_migrations(connection):
    context.configure(connection=connection, target_metadata=target_metadata)
    with context.begin_transaction():
        context.run_migrations()


async def run_async_migrations() -> None:
    """In this scenario we need to create an AsyncEngine
    and associate a connection with the context.
    """
    connectable = async_engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    async with connectable.connect() as connection:
        await connection.run_sync(do_run_migrations)

    await connectable.dispose()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode, bridging Alembic's sync-only
    migration context to our async engine (db.py uses AsyncEngine/AsyncSession
    throughout — see #41 for why the Postgres layer is async).

    asyncio.run() raises RuntimeError if called from inside an already-running
    event loop. This happens under `uvicorn app.main:app` (the actual CLI
    invocation used in Dockerfile/docker-compose) — uvicorn's app-loading path
    has a loop active by the time this module executes, unlike a bare
    `python3 -c "import app.main"`. Detect that case and run the coroutine on
    a fresh event loop in a separate thread instead.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        asyncio.run(run_async_migrations())
    else:
        import threading

        exception_holder = []

        def _run_in_thread():
            try:
                asyncio.run(run_async_migrations())
            except Exception as e:
                exception_holder.append(e)

        thread = threading.Thread(target=_run_in_thread)
        thread.start()
        thread.join()
        if exception_holder:
            raise exception_holder[0]


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
