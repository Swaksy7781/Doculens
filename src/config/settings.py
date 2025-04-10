"""
Application configuration settings
Contains centralized settings for the PDF Chat application
"""

import os
from pathlib import Path

# Environment variable name for Google API key
GOOGLE_API_KEY_ENV = "GOOGLE_API_KEY"

# API models and parameters
LLM_MODEL = "gemini-1.5-pro"  # Google's Gemini 1.5 Pro model
EMBEDDING_MODEL = "models/embedding-001"  # Google's embedding model
LLM_TEMPERATURE = 0.2  # Lower for more deterministic outputs
LLM_MAX_OUTPUT_TOKENS = 2048  # Maximum output size
LLM_TOP_K = 40  # Top k tokens to consider during generation
LLM_TOP_P = 0.8  # Top p tokens to consider during generation

# Application metadata
APP_TITLE = "PDF Chat"
APP_ICON = "📚"
APP_VERSION = "1.0.0"
APP_LAST_UPDATED = "2023-04-06"

# Security settings
INPUT_MAX_LENGTH = 4000  # Maximum input length for query to prevent DOS
RATE_LIMIT_QUERIES_PER_MINUTE = 10  # Maximum number of queries per minute
RATE_LIMIT_QUERIES_PER_HOUR = 100  # Maximum number of queries per hour

# PDF processing settings
MAX_PDF_SIZE_MB = 100  # Maximum size of each PDF file
MAX_PDF_COUNT = 10  # Maximum number of PDFs in one batch
CHUNK_SIZE = 1000  # Size of chunks for text splitting
CHUNK_OVERLAP = 200  # Overlap between chunks

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
VECTOR_STORE_PATH = os.path.join(BASE_DIR, "faiss_index")
LOG_DIR = os.path.join(BASE_DIR)
APP_LOG_FILE = os.path.join(LOG_DIR, "app.log")
AUDIT_LOG_FILE = os.path.join(LOG_DIR, "audit.log")
SECURITY_LOG_FILE = os.path.join(LOG_DIR, "security.log")

# Create directories if they don't exist
os.makedirs(VECTOR_STORE_PATH, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Data retention options
RETENTION_OPTIONS = [
    "Session Only (Default)",
    "1 Hour",
    "1 Day",
    "1 Week"
]