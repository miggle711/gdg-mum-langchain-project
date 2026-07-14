from pathlib import Path

from pydantic_settings import BaseSettings

# pydantic-settings resolves a relative env_file against the process's
# current working directory, not this file's location — so "backend/.env"
# would silently load a different (or missing) file depending on whether
# the app is launched from the repo root or from backend/. Anchoring to
# __file__ makes the resolved path independent of invocation directory.
_BACKEND_DIR = Path(__file__).resolve().parent.parent
_ENV_FILE = _BACKEND_DIR / ".env"


class Settings(BaseSettings):
    google_api_key: str
    redis_url: str = "redis://localhost:6379"
    elasticsearch_url: str = "http://localhost:9200"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:5432/ecommerce"
    search_cache_ttl_seconds: int = 60 * 60 * 6
    search_cache_similarity_threshold: float = 0.92
    conversation_ttl_seconds: int = 60 * 60 * 24
    conversation_summary_threshold: int = 20  # messages (10 turns) before summarising
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    class Config:
        env_file = str(_ENV_FILE)
        extra = "ignore"


settings = Settings()
