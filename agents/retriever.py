"""Retriever Agent Node."""
import logging
from typing import Dict, Any
from utils.chroma_store import ChromaStoreManager
from config import DEFAULT_TOP_K

logger = logging.getLogger(__name__)


def retriever_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to retrieve top-K relevant chunks from ChromaDB vector store."""
    rewritten_query = state.get("rewritten_question") or state.get("question", "")
    top_k = state.get("top_k", DEFAULT_TOP_K)
    processing_steps = list(state.get("processing_steps", []))

    try:
        store_manager = ChromaStoreManager()
        chunks = store_manager.retrieve(query=rewritten_query, top_k=top_k)

        step_info = {
            "name": "Retriever",
            "status": "success",
            "detail": f"Retrieved {len(chunks)} chunks from ChromaDB",
        }
        processing_steps.append(step_info)
        logger.info("Retrieved %d chunks for query '%s'", len(chunks), rewritten_query)

        return {
            **state,
            "retrieved_chunks": chunks,
            "processing_steps": processing_steps,
        }

    except Exception as e:
        logger.error("Retriever node failed: %s", str(e))
        step_info = {
            "name": "Retriever",
            "status": "error",
            "detail": f"Retrieval failed: {str(e)}",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "retrieved_chunks": [],
            "processing_steps": processing_steps,
        }
