from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=("../.env", ".env"), env_file_encoding="utf-8", extra="ignore"
    )

    app_name: str = "Wizarding Library RAG"
    app_env: str = "development"
    log_level: str = "INFO"
    skip_pipeline_startup: bool = False
    frontend_origins: str = "http://localhost:8501,http://127.0.0.1:8501"

    qdrant_path: Path = Path("backend/data/vector_store")
    qdrant_url: str | None = None
    qdrant_api_key: str | None = None
    qdrant_collection: str = "harry_potter_books"
    embedding_backend: str = "hashing"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    hashing_dimensions: int = Field(default=768, ge=128, le=4096)
    top_k: int = Field(default=4, ge=1, le=10)
    min_relevance_score: float = Field(default=0.25, ge=0, le=1)

    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "llama3.2:3b"
    allow_extractive_fallback: bool = True

    @property
    def cors_origins(self) -> list[str]:
        return [item.strip() for item in self.frontend_origins.split(",") if item.strip()]

    @field_validator("qdrant_path", mode="after")
    @classmethod
    def resolve_project_path(cls, value: Path) -> Path:
        if value.is_absolute():
            return value
        project_root = Path(__file__).resolve().parents[3]
        return (project_root / value).resolve()


@lru_cache
def get_settings() -> Settings:
    return Settings()
