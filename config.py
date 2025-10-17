# config.py
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache

# Define the structure for all required secrets/config
class Settings(BaseSettings):
    # API Keys/Tokens
    GEMINI_API_KEY: str
    GITHUB_TOKEN: str
    
    # Project-specific variables
    STUDENT_SECRET: str
    GITHUB_USERNAME: str
    
    # Define which file to load settings from
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

# Use lru_cache to load the settings only once, improving performance
@lru_cache()
def get_settings():
    """Returns the cached settings object."""
    return Settings()