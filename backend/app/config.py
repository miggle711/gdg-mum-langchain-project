from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str
    redis_url: str = "redis://localhost:6379"
    elasticsearch_url: str = "http://localhost:9200"
    search_cache_ttl_seconds: int = 60 * 60 * 6
    search_cache_similarity_threshold: float = 0.92
    conversation_ttl_seconds: int = 60 * 60 * 24
    conversation_summary_threshold: int = 20  # messages (10 turns) before summarising
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "gdg-mum-langchain-project"
    langfuse_public_key: str = ""
    langfuse_secret_key: str = ""
    langfuse_base_url: str = "https://cloud.langfuse.com"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
