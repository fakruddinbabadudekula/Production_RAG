from pydantic_settings import (
    BaseSettings,
)  # Automatically reads environment variables and Validates the data
from pydantic import Field  # Add additional information like max,min,description..
from functools import (
    lru_cache,
)  # to cache the function for same parameters, loads first time then cached.


class Settings(BaseSettings):

    # API keys....
    GOOGLE_API_KEY: str
    OPENROUTER_BASE_URL: str
    OPENROUTER_API_KEY: str

    # App Information
    APP_NAME: str = "NotebookLm"

    # Model Config
    CURRENT_CHAT_MODEL: str = "openai/gpt-oss-20b:free"
    TEMPERATURE: float = 0.7
    
    # Vector 
    VECTOR_FOLDER:str="vectors"

    class Config:
        env_file = ".env"  # Look for .env file
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
