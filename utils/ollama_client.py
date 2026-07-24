"""Ollama client wrapper for local LLM and Embedding models."""
import logging
from typing import Optional, List
import requests

from llama_index.llms.ollama import Ollama
from llama_index.embeddings.ollama import OllamaEmbedding
from config import OLLAMA_BASE_URL, LLM_MODEL, EMBEDDING_MODEL

logger = logging.getLogger(__name__)


def check_ollama_status(base_url: str = OLLAMA_BASE_URL) -> bool:
    """Check if Ollama service is running locally."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        return response.status_code == 200
    except Exception:
        return False


def get_available_models(base_url: str = OLLAMA_BASE_URL) -> List[str]:
    """Retrieve list of pulled Ollama models."""
    try:
        response = requests.get(f"{base_url}/api/tags", timeout=3)
        if response.status_code == 200:
            data = response.json()
            return [m.get("name", "") for m in data.get("models", [])]
    except Exception as e:
        logger.warning("Could not fetch Ollama models: %s", str(e))
    return []


def get_ollama_llm(
    model_name: str = LLM_MODEL,
    temperature: float = 0.1,
    base_url: str = OLLAMA_BASE_URL,
    request_timeout: float = 360.0,
) -> Ollama:
    """Instantiate a LlamaIndex Ollama LLM instance with 360s timeout for local inference."""
    return Ollama(
        model=model_name,
        base_url=base_url,
        temperature=temperature,
        request_timeout=request_timeout,
    )


def get_ollama_embedding(
    model_name: str = EMBEDDING_MODEL,
    base_url: str = OLLAMA_BASE_URL,
) -> OllamaEmbedding:
    """Instantiate a LlamaIndex OllamaEmbedding instance."""
    return OllamaEmbedding(
        model_name=model_name,
        base_url=base_url,
    )
