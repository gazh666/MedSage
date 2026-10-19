import os
import json
from urllib.parse import urlparse, parse_qsl, urlencode, urlunparse
from typing import List, Union, Any
from pydantic import field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # --- Basic App Info ---
    ENVIRONMENT: str = "production"
    PROJECT_NAME: str = "Rural AI Doctor"
    VERSION: str = "1.1.0"
    API_V1_STR: str = "/api/v1"
    DEBUG: bool = False  
    
    # --- Infrastructure ---

    DATABASE_URL: str = "postgresql+psycopg://postgres:postgres@127.0.0.1:5434/rural_ai_doctor"
    REDIS_URL: str = "memory://" 
    
    # --- Google AI & Vertex AI ---
    # Updated to the current production standard while keeping your location
    GOOGLE_API_KEY: str = ""
    NCBI_API_KEY: str = ""
    OPENFDA_API_KEY: str = ""
    GOOGLE_CLOUD_PROJECT: str = "" 
    VERTEX_AI_LOCATION: str = "us-central1"
    GOOGLE_APPLICATION_CREDENTIALS: str = ""
    GEMINI_MODEL: str = 'gemini-3.1-flash-lite-preview'

    # --- Pluggable LLM Provider（OpenAI 兼容：通义千问 / DeepSeek / 智谱 GLM）---
    # 设 LLM_PROVIDER=openai 即切换为国内 OpenAI 兼容模型，无需翻墙。
    LLM_PROVIDER: str = "gemini"  # gemini | openai
    LLM_API_KEY: str = ""
    LLM_BASE_URL: str = ""
    LLM_MODEL: str = "" 
    
    # UPDATED: Use the new multimodal embedding model
    EMBEDDING_MODEL: str = "models/text-embedding-004"
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    TOP_K_RESULTS: int = 3
    RAG_MAX_UPLOAD_MB: int = 40
    RAG_MAX_CHUNKS_PER_DOC: int = 120

    # --- Lung Sound Model (PyTorch) ---
    # 训练好的 PBRNet 权重（含 'model' 键的 checkpoint dict）。
    MODEL_LUNG_SOUND_CHECKPOINT_PATH: str = "checkpoint/lung_best.pt"
    # RSC-FTF 训练代码根目录（含 models/ 与 BEATs.py），用于复用模型架构。
    MODEL_LUNG_SOUND_RSC_ROOT: str = ""
    
    # --- Monitoring & Observability ---
    LANGCHAIN_TRACING_V2: bool = False # Default to False to save on API usage
    LANGCHAIN_API_KEY: str = "" 
    LANGCHAIN_PROJECT: str = "rural-ai-doctor"
    SENTRY_DSN: str = ""  

    # --- Security ---
    SECRET_KEY: str = "your-secret-key-change-this-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 60 # Increased for better UX
    
    # --- CORS Configuration ---
    BACKEND_CORS_ORIGINS: List[str] = [
        "http://localhost:3000",
        "https://rural-ai-doctor-rwp9-j5kvvx8u8-rockys-projects-e671580a.vercel.app"
    ]

    @field_validator("DATABASE_URL", mode="before")
    @classmethod
    def assemble_db_url(cls, v: str) -> str:
        """Normalize Postgres URLs to an AsyncEngine-compatible driver."""
        if isinstance(v, str):
            if v.startswith("postgres://"):
                v = v.replace("postgres://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql://"):
                v = v.replace("postgresql://", "postgresql+asyncpg://", 1)
            elif v.startswith("postgresql+psycopg://"):
                v = v.replace("postgresql+psycopg://", "postgresql+asyncpg://", 1)

            # PgBouncer compatibility: disable prepared statement cache.
            parsed = urlparse(v)
            query_params = dict(parse_qsl(parsed.query, keep_blank_values=True))
            # asyncpg expects `ssl`, not `sslmode`.
            if "sslmode" in query_params and "ssl" not in query_params:
                query_params["ssl"] = query_params["sslmode"]
            query_params.pop("sslmode", None)
            query_params.setdefault("prepared_statement_cache_size", "0")
            v = urlunparse(parsed._replace(query=urlencode(query_params)))
        return v

    @field_validator("BACKEND_CORS_ORIGINS", mode="before")
    @classmethod
    def assemble_cors_origins(cls, v: Any) -> Union[List[str], str]:
        """Handles strings, comma-separated lists, and JSON arrays from ENV."""
        if isinstance(v, str) and not v.startswith("["):
            return [i.strip() for i in v.split(",")]
        elif isinstance(v, str) and v.startswith("["):
            return json.loads(v)
        return v

    # --- Rate Limiting ---
    RATE_LIMIT_PER_MINUTE: int = 100

    # --- Email Configuration ---
    MAIL_USERNAME: str = ""
    MAIL_PASSWORD: str = ""
    MAIL_FROM: str = "noreply@rural-ai.doc"
    MAIL_PORT: int = 587
    MAIL_SERVER: str = "smtp.gmail.com"
    MAIL_STARTTLS: bool = True
    MAIL_SSL_TLS: bool = False
    FRONTEND_URL: str = "http://localhost:3000"
    EMERGENCY_ALERT_EMAIL: str = ""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

settings = Settings()
