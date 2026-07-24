"""Answer Generator Agent Node."""
import logging
from typing import Dict, Any, List
from utils.helpers import load_prompt
from utils.ollama_client import get_ollama_llm
from config import LLM_MODEL, DEFAULT_TEMPERATURE

logger = logging.getLogger(__name__)


def answer_generator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to synthesize an answer from graded relevant chunks."""
    question = state.get("question", "")
    graded_chunks = state.get("graded_chunks", [])
    if not graded_chunks:
        # Fallback to retrieved chunks if grading removed all
        graded_chunks = state.get("retrieved_chunks", [])

    processing_steps = list(state.get("processing_steps", []))
    temperature = state.get("temperature", DEFAULT_TEMPERATURE)

    if not graded_chunks:
        step_info = {
            "name": "Answer Generator",
            "status": "warning",
            "detail": "No relevant context passages found in documents.",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "answer": "The uploaded documents do not contain relevant information to answer this question.",
            "processing_steps": processing_steps,
        }

    # Format context passages cleanly
    context_passages: List[str] = []
    for idx, c in enumerate(graded_chunks):
        fname = c.get("filename", "document.pdf")
        page = c.get("page_number", 1)
        text = c.get("text", "")
        context_passages.append(f"[Source {idx+1}: {fname} (Page {page})]\n{text}")

    context_str = "\n\n---\n\n".join(context_passages)
    prompt_template = load_prompt("answer.md")

    # Safe string replacement without str.format brace errors
    formatted_prompt = (
        prompt_template
        .replace("{question}", str(question))
        .replace("{context_str}", str(context_str))
    )

    try:
        llm = get_ollama_llm(model_name=LLM_MODEL, temperature=temperature)
        response = llm.complete(formatted_prompt)
        answer_text = response.text.strip()

        step_info = {
            "name": "Answer Generator",
            "status": "success",
            "detail": "Synthesized final answer from document context",
        }
        processing_steps.append(step_info)
        logger.info("Generated answer successfully (%d chars)", len(answer_text))

        return {
            **state,
            "answer": answer_text,
            "processing_steps": processing_steps,
        }

    except Exception as e:
        logger.error("Answer generation failed: %s", str(e), exc_info=True)
        step_info = {
            "name": "Answer Generator",
            "status": "error",
            "detail": f"Failed to generate answer: {str(e)}",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "answer": f"An error occurred while generating the answer: {str(e)}. Please verify Ollama is running.",
            "processing_steps": processing_steps,
        }
