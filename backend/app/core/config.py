"""
Configuration management for RAG system
"""

import os
import yaml
from typing import List, Optional
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Settings(BaseSettings):
    """Application settings"""
    
    # API Configuration
    api_host: str = os.getenv("API_HOST", "0.0.0.0")
    api_port: int = int(os.getenv("API_PORT", "8000"))
    
    # Ollama Configuration
    ollama_base_url: str = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    ollama_model: str = os.getenv("OLLAMA_MODEL", "llama2")
    ollama_embedding_model: str = os.getenv("OLLAMA_EMBEDDING_MODEL", "llama2")
    ollama_temperature: float = 0.7
    ollama_max_tokens: int = 2048
    
    # ChromaDB Configuration
    chroma_persist_directory: str = os.getenv("CHROMA_PERSIST_DIRECTORY", "./data/vector_db")
    chroma_collection_name: str = "rag_documents"
    chroma_distance_function: str = "cosine"
    
    # Document Processing
    chunk_size: int = 1000
    chunk_overlap: int = 200
    max_file_size_mb: int = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
    allowed_extensions: List[str] = [".pdf", ".docx", ".pptx", ".txt", ".md"]
    upload_dir: str = os.getenv("UPLOAD_DIR", "./data/uploads")
    
    # Retrieval Configuration
    retrieval_top_k: int = 5
    similarity_threshold: float = 0.7
    
    # CORS Configuration
    cors_origins: List[str] = ["http://localhost:8501", "http://127.0.0.1:8501"]
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    
    class Config:
        env_file = ".env"

def load_config_from_yaml(config_path: str = "backend/config.yaml") -> dict:
    """Load configuration from YAML file"""
    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)
    except FileNotFoundError:
        print(f"Config file {config_path} not found, using default settings")
        return {}

# Global settings instance
settings = Settings()

# Load YAML config and update settings if available
yaml_config = load_config_from_yaml()
if yaml_config:
    # Update settings with YAML values
    for section, values in yaml_config.items():
        if isinstance(values, dict):
            for key, value in values.items():
                attr_name = f"{section}_{key}" if section != "api" else key
                if hasattr(settings, attr_name):
                    setattr(settings, attr_name, value)
