from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    google_api_key: str
    database_url: str = "postgresql://gdg:gdg@localhost:5432/ecommerce"
    redis_url: str = "redis://localhost:6379"
    elasticsearch_url: str = "http://localhost:9200"
    search_cache_ttl_seconds: int = 60 * 60 * 6
    search_cache_similarity_threshold: float = 0.92
    conversation_ttl_seconds: int = 60 * 60 * 24
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "gdg-mum-langchain-project"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
