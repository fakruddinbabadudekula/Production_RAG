from pydantic_settings import (
    BaseSettings,
)  # Automatically reads environment variables and Validates the data
from pydantic import Field  # Add additional information like max,min,description..
from functools import (
    lru_cache,
)  # to cache the function for same parameters, loads first time then cached.

from pathlib import Path

from regex import P


class Settings(BaseSettings):

    # API keys....
    OPENROUTER_BASE_URL: str
    OPENROUTER_API_KEY: str

    # Model Config
    CURRENT_CHAT_MODEL: str = "openai/gpt-oss-20b:free"
    TEMPERATURE: float = 0.7

    # Base File Path
    BASE_PATH: Path = Path(__file__).resolve().parents[2]

    # App Information
    APP_NAME: str = "NotebookLm"
    APP_PATH: Path = BASE_PATH / "app"

    # Storage Path
    STORAGE_PATH: Path = BASE_PATH / "storage"

    # Vector
    VECTOR_FOLDER: Path = STORAGE_PATH / "vectors"

    # Data Path where uploaded files are stored
    FILE_UPLOAD_PATH: Path = STORAGE_PATH / "upload_files"

    # Embedding Model
    EMBED_MODEL: str = "sentence-transformers/all-MiniLM-L6-v2"
    EMBED_MODEL_SIZE: int = 384

    # DataBase
    DATABASE_URL:str
    
    # Password Hashing
    HASHING_ALGO:str="argon2"
    
    # Authentication
    ACCESS_TOKEN_EXPIRE_MINUTES:int
    REFRESH_TOKEN_EXPIRE_DAYS:int
    SECRET_KEY:str
    ALGORITHM:str
    
    
    # Timeouts
    CHAT_MODEL_TIMEOUT: int = 30
    LLM_CALL_ASYNC_TIMEOUT: int = 40

    # Retries
    MAX_LLM_CALL_RETRIES: int = 3
    MAX_PDF_PROCESS_RETRY: int = 3
    class Config:
        env_file = ".env"  # Look for .env file
        case_sensitive = True
        extra = "ignore"



@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
