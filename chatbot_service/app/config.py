from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # API identity
    app_title: str = "Chatbot Service"
    app_version: str = "0.1.0"
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    # Always 1 — GPU VRAM cannot be shared across uvicorn worker processes.
    # Scale horizontally with multiple containers, not multiple workers.
    api_workers: int = 1

    # Environment
    env: Literal["dev", "prod"] = "dev"

    # Model
    model_mode: Literal["mock", "real"] = "mock"
    model_path: str = "/models/adapter"
    base_model_name: str = "Qwen/Qwen3-4B"

    # Generation — temperature sourced from Colab notebook test cell
    max_new_tokens: int = 512
    temperature: float = 0.3
    do_sample: bool = True

    # Logging
    log_level: str = "INFO"
    log_format: Literal["text", "json"] = "text"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )


settings = Settings()
