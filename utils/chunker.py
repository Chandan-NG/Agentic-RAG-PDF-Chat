"""Semantic text chunking module using LlamaIndex SentenceSplitter."""
import logging
from typing import List
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import BaseNode, Document

logger = logging.getLogger(__name__)


def chunk_documents(
    documents: List[Document],
    chunk_size: int = 512,
    chunk_overlap: int = 50,
) -> List[BaseNode]:
    """Chunk LlamaIndex documents into semantic text nodes.

    Args:
        documents: List of page-level LlamaIndex Documents.
        chunk_size: Target token/character chunk size.
        chunk_overlap: Overlapping characters between adjacent chunks.

    Returns:
        List of LlamaIndex BaseNode objects containing chunk text and metadata.
    """
    if not documents:
        return []

    splitter = SentenceSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    nodes = splitter.get_nodes_from_documents(documents)

    for idx, node in enumerate(nodes):
        # Guarantee all metadata keys are accessible in metadata
        node.metadata["chunk_id"] = f"{node.metadata.get('document_id', 'doc')}_c{idx+1}"
        if "filename" not in node.metadata and hasattr(node, "extra_info"):
            node.metadata["filename"] = node.extra_info.get("filename", "unknown.pdf")

    logger.info("Generated %d chunks from %d document pages", len(nodes), len(documents))
    return nodes
