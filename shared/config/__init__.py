"""Centralized configuration management using pydantic-settings.

All services import `get_settings()` rather than reading os.environ
directly, so configuration stays typed, validated, and testable.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # General
    environment: str = "development"
    log_level: str = "INFO"
    service_name: str = "voiceops-api"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Postgres
    postgres_host: str = "localhost"
    postgres_port: int = 5432
    postgres_db: str = "voiceops"
    postgres_user: str = "voiceops"
    postgres_password: str = "voiceops"

    # Redis
    redis_host: str = "localhost"
    redis_port: int = 6379

    # Kafka
    kafka_bootstrap_servers: str = "localhost:9092"

    # Object storage (S3 / MinIO)
    s3_endpoint_url: str | None = "http://localhost:9000"
    s3_bucket: str = "voiceops-audio"
    s3_access_key: str = "minioadmin"
    s3_secret_key: str = "minioadmin"

    # Vector DB
    qdrant_host: str = "localhost"
    qdrant_port: int = 6333

    # MLflow
    mlflow_tracking_uri: str = "http://localhost:5000"

    # ASR
    asr_model_name: str = "openai/whisper-base"
    asr_device: str = "cpu"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql+asyncpg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
