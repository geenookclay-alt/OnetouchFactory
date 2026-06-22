from pydantic_settings import BaseSettings
from pathlib import Path

class Settings(BaseSettings):
    apify_token: str = ''
    gemini_api_key: str = ''
    nexlev_api_key: str = ''
    notion_token: str = ''
    ollama_host: str = 'http://127.0.0.1:11434'
    ollama_model: str = 'qwen2.5:14b'
    embed_model: str = 'nomic-embed-text'
    sqlite_path: str = './db/discover.db'
    qdrant_path: str = './db/qdrant'
    host: str = '0.0.0.0'
    port: int = 8000

    class Config:
        env_file = Path(__file__).parent.parent / '.env'

settings = Settings()
