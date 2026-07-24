"""Citation Formatter Agent Node."""
import logging
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


def citation_formatter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to format source citations and snippets."""
    graded_chunks = state.get("graded_chunks", [])
    if not graded_chunks:
        graded_chunks = state.get("retrieved_chunks", [])

    processing_steps = list(state.get("processing_steps", []))

    sources: List[Dict[str, Any]] = []
    seen_sources = set()

    for chunk in graded_chunks:
        fname = chunk.get("filename", "unknown.pdf")
        page = chunk.get("page_number", 1)
        key = (fname, page)

        if key in seen_sources:
            continue
        seen_sources.add(key)

        snippet = chunk.get("text", "").strip()
        if len(snippet) > 280:
            snippet = snippet[:280] + "..."

        sources.append({
            "filename": fname,
            "page_number": page,
            "snippet": snippet,
            "score": chunk.get("score", 0.0),
        })

    step_info = {
        "name": "Citation Formatter",
        "status": "success",
        "detail": f"Formatted {len(sources)} source citations",
    }
    processing_steps.append(step_info)
    logger.info("Formatted %d citations", len(sources))

    return {
        **state,
        "sources": sources,
        "processing_steps": processing_steps,
    }
