"""Query Rewriter Agent Node."""
import logging
from typing import Dict, Any
from utils.helpers import load_prompt
from utils.ollama_client import get_ollama_llm
from config import LLM_MODEL

logger = logging.getLogger(__name__)


def query_rewriter_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to rewrite user query for vector similarity retrieval."""
    question = state.get("question", "")
    retry_count = state.get("retry_count", 0)
    processing_steps = list(state.get("processing_steps", []))
    previous_queries = state.get("rewritten_question", "")

    try:
        prompt_template = load_prompt("rewrite.md")
        # Safe replacement without str.format brace errors
        formatted_prompt = (
            prompt_template
            .replace("{question}", str(question))
            .replace("{previous_queries}", str(previous_queries) if retry_count > 0 else "None")
        )

        llm = get_ollama_llm(model_name=LLM_MODEL, temperature=0.2)
        response = llm.complete(formatted_prompt)
        rewritten = response.text.strip().replace('"', '').replace("'", "")

        if not rewritten:
            rewritten = question

        step_info = {
            "name": "Query Rewrite",
            "status": "success",
            "detail": f"Rewrote query (Attempt {retry_count + 1}): '{rewritten}'",
        }
        processing_steps.append(step_info)
        logger.info("Query rewritten from '%s' to '%s'", question, rewritten)

        return {
            **state,
            "rewritten_question": rewritten,
            "processing_steps": processing_steps,
        }

    except Exception as e:
        logger.error("Query rewrite node failed: %s", str(e), exc_info=True)
        step_info = {
            "name": "Query Rewrite",
            "status": "warning",
            "detail": f"Fallback to original question due to error: {str(e)}",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "rewritten_question": question,
            "processing_steps": processing_steps,
        }
