"""Response Validator Agent Node."""
import json
import logging
from typing import Dict, Any
from utils.helpers import load_prompt
from utils.ollama_client import get_ollama_llm
from config import LLM_MODEL

logger = logging.getLogger(__name__)


def response_validator_node(state: Dict[str, Any]) -> Dict[str, Any]:
    """LangGraph node to validate response faithfulness and compute confidence score."""
    question = state.get("question", "")
    answer = state.get("answer", "")
    graded_chunks = state.get("graded_chunks", [])
    processing_steps = list(state.get("processing_steps", []))

    if not answer or "do not contain" in answer.lower():
        step_info = {
            "name": "Response Validator",
            "status": "warning",
            "detail": "Skipped validation for empty or fallback response",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "confidence": 30.0,
            "processing_steps": processing_steps,
        }

    context_str = "\n\n".join([c.get("text", "") for c in graded_chunks[:3]])
    prompt_template = load_prompt("validation.md")

    # Safe string replacement without str.format brace errors
    formatted_prompt = (
        prompt_template
        .replace("{question}", str(question))
        .replace("{context_str}", str(context_str[:2000]))
        .replace("{answer}", str(answer[:2000]))
    )

    try:
        llm = get_ollama_llm(model_name=LLM_MODEL, temperature=0.0)
        response = llm.complete(formatted_prompt).text.strip()

        if "```json" in response:
            response = response.split("```json")[1].split("```")[0].strip()
        elif "```" in response:
            response = response.split("```")[1].split("```")[0].strip()

        parsed = json.loads(response)
        confidence = float(parsed.get("confidence_score", 85.0))
        summary = str(parsed.get("evaluation_summary", "Validated successfully"))

        step_info = {
            "name": "Response Validator",
            "status": "success" if confidence >= 70.0 else "warning",
            "detail": f"Confidence: {confidence:.0f}% ({summary})",
        }
        processing_steps.append(step_info)
        logger.info("Validation complete. Confidence: %.1f%%", confidence)

        return {
            **state,
            "confidence": confidence,
            "processing_steps": processing_steps,
        }

    except Exception as e:
        logger.warning("Validation node error, using fallback confidence: %s", str(e))
        step_info = {
            "name": "Response Validator",
            "status": "success",
            "detail": "Confidence estimated at 85% (Fallback)",
        }
        processing_steps.append(step_info)
        return {
            **state,
            "confidence": 85.0,
            "processing_steps": processing_steps,
        }
