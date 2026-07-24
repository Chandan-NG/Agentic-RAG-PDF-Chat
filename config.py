"""Configuration settings and design system tokens for Agentic RAG PDF Chat."""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
UPLOADS_DIR = BASE_DIR / "uploads"
CHROMA_DIR = BASE_DIR / "chroma_db"
PROMPTS_DIR = BASE_DIR / "prompts"

# Ensure required directories exist
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_DIR.mkdir(parents=True, exist_ok=True)
PROMPTS_DIR.mkdir(parents=True, exist_ok=True)

# Ollama Models & API Configuration
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2:7b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "nomic-embed-text")

# ChromaDB Configuration
CHROMA_PERSIST_DIR = str(CHROMA_DIR)
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "pdf_collection")

# Default RAG Parameters
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.1"))
MAX_RETRIES = int(os.getenv("MAX_RETRIES", "2"))

# Linear Design System Color Tokens (Strictly from DESIGN.md)
COLOR_CANVAS = "#010102"
COLOR_SURFACE_1 = "#0e0f12"
COLOR_SURFACE_2 = "#16171d"
COLOR_SURFACE_3 = "#1f2129"
COLOR_SURFACE_4 = "#282a36"

COLOR_HAIRLINE = "#23252a"
COLOR_HAIRLINE_STRONG = "#343842"
COLOR_HAIRLINE_TERTIARY = "#1c1e24"

COLOR_INK = "#f7f8f8"
COLOR_INK_MUTED = "#d0d6e0"
COLOR_INK_SUBTLE = "#8a8f98"
COLOR_INK_TERTIARY = "#62666d"

COLOR_PRIMARY = "#5e6ad2"
COLOR_PRIMARY_HOVER = "#828fff"
COLOR_PRIMARY_FOCUS = "#5e69d1"

COLOR_SEMANTIC_SUCCESS = "#27a644"
