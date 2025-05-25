"""
Logging configuration for RAG system
"""

import logging
import sys
from pathlib import Path
from backend.app.core.config import settings

def setup_logger(name: str = "rag_system") -> logging.Logger:
    """Setup logger with configured format and level"""
    
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, settings.log_level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Create formatter
    formatter = logging.Formatter(settings.log_format)
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler
    log_dir = Path("logs")
    log_dir.mkdir(exist_ok=True)
    file_handler = logging.FileHandler(log_dir / "rag_system.log")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    return logger

# Global logger instance
logger = setup_logger()
