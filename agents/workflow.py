"""LangGraph Workflow definition for Agentic RAG PDF Chat."""
import logging
from typing import TypedDict, List, Dict, Any
from langgraph.graph import StateGraph, END, START

from agents.query_rewriter import query_rewriter_node
from agents.retriever import retriever_node
from agents.relevance_grader import relevance_grader_node
from agents.answer_generator import answer_generator_node
from agents.response_validator import response_validator_node
from agents.citation_formatter import citation_formatter_node
from config import MAX_RETRIES

logger = logging.getLogger(__name__)


class GraphState(TypedDict):
    """Schema for the LangGraph state."""
    question: str
    rewritten_question: str
    retrieved_chunks: List[Dict[str, Any]]
    graded_chunks: List[Dict[str, Any]]
    confidence: float
    retry_count: int
    max_retries: int
    top_k: int
    temperature: float
    answer: str
    sources: List[Dict[str, Any]]
    processing_steps: List[Dict[str, Any]]


def decide_retrieval_quality(state: GraphState) -> str:
    """Conditional decision node to evaluate retrieval quality and trigger rewrite loops if needed."""
    graded_chunks = state.get("graded_chunks", [])
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", MAX_RETRIES)

    # If no relevant chunks found and under max retries, loop back to query rewriter
    if not graded_chunks and retry_count < max_retries:
        logger.info("Retrieval quality poor. Retrying query rewrite (%d/%d)", retry_count + 1, max_retries)
        return "rewrite_again"

    logger.info("Proceeding to answer generator")
    return "generate_answer"


def increment_retry_node(state: GraphState) -> GraphState:
    """Helper node to increment retry counter before query rewrite loop."""
    current_retries = state.get("retry_count", 0)
    return {
        **state,
        "retry_count": current_retries + 1,
    }


def create_agentic_rag_graph():
    """Build and compile the real LangGraph workflow for Agentic RAG."""
    workflow = StateGraph(GraphState)

    # Add Nodes
    workflow.add_node("query_rewriter", query_rewriter_node)
    workflow.add_node("retriever", retriever_node)
    workflow.add_node("relevance_grader", relevance_grader_node)
    workflow.add_node("increment_retry", increment_retry_node)
    workflow.add_node("answer_generator", answer_generator_node)
    workflow.add_node("response_validator", response_validator_node)
    workflow.add_node("citation_formatter", citation_formatter_node)

    # Define Fixed Edges
    workflow.add_edge(START, "query_rewriter")
    workflow.add_edge("query_rewriter", "retriever")
    workflow.add_edge("retriever", "relevance_grader")

    # Conditional Routing from Relevance Grader
    workflow.add_conditional_edges(
        "relevance_grader",
        decide_retrieval_quality,
        {
            "rewrite_again": "increment_retry",
            "generate_answer": "answer_generator",
        },
    )

    workflow.add_edge("increment_retry", "query_rewriter")
    workflow.add_edge("answer_generator", "response_validator")
    workflow.add_edge("response_validator", "citation_formatter")
    workflow.add_edge("citation_formatter", END)

    app = workflow.compile()
    return app
