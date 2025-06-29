import os
from dataclasses import dataclass, field
from typing import Dict
from pathlib import Path

@dataclass
class Config:
    # API Key for Gemini (used in sports performance assistant)
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "AIzaSyA9aIFmIQrov6ls9FrAS0YvGG7wev5OMIY")
    
    # File paths
        VECTOR_DB_PATH: str = "./vector_db"
    
    # Countries and their language codes
        
    # Model configurations
    EMBEDDING_MODEL: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    
    # Gemini Model
    GEMINI_MODEL: str = "gemini-2.0-flash-exp"  # Updated to 2.0 Flash
    
    # Processing parameters
    CHUNK_SIZE: int = 1000
    CHUNK_OVERLAP: int = 200
    MAX_TOKENS: int = 4000
    TEMPERATURE: float = 0.1
    
    # Vector DB settings
    SIMILARITY_THRESHOLD: float = 0.7
    MAX_RESULTS: int = 5
    
    # Language detection confidence threshold
    COUNTRY_DETECTION_THRESHOLD: float = 0.6

# Global config instance for performance use case
config = Config()

def validate_config() -> bool:
    """Validate that required configurations are set"""
    if config.GOOGLE_API_KEY.startswith("your-"):
        print("⚠️  Warning: Gemini API key is not configured")
        return False
    
    # Check if civil codes directory exists
        
    return True

        raise ValueError(f"Country '{country}' not supported")
    
    filename = config.COUNTRIES[country]["file"]
    return os.path.join(config.CIVIL_CODES_DIR, filename)

        raise ValueError(f"Country '{country}' not supported")
    
    return config.COUNTRIES[country]["lang_code"]