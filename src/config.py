from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

_BASE_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=_BASE_DIR / ".env", env_file_encoding="utf-8", extra="ignore")

    mlflow_tracking_uri: str
    mlflow_artifact_store: str

    log_file: str

    database_url: str

    openai_api_key: str
    openai_base_url: str
    openai_embedding_model: str
    openai_embedding_dimensions: int
    openai_chat_model: str
    openai_chat_model_temperature: float

    retrieval_top_k: int


@lru_cache
def get_settings():
    return Settings()  # type: ignore
