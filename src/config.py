from pydantic_settings import BaseSettings
from typing import Optional
import os
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # GitHub Configuration
    github_token: str = os.getenv("GITHUB_TOKEN", "")
    
    # API Configuration
    cache_duration_minutes: int = int(os.getenv("CACHE_DURATION_MINUTES", "30"))
    max_retries: int = int(os.getenv("MAX_RETRIES", "3"))
    backoff_factor: int = int(os.getenv("BACKOFF_FACTOR", "2"))
    
    # LLM Configuration
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY")
    
    class Config:
        env_file = ".env"

settings = Settings() 