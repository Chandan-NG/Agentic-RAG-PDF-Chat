"""ChromaDB vector store management, stats calculation, and retriever wrapper."""
import logging
from typing import Dict, List, Any, Optional
import chromadb

from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.core.schema import BaseNode
from llama_index.core.retrievers import VectorIndexRetriever

from config import CHROMA_PERSIST_DIR, COLLECTION_NAME, DEFAULT_TOP_K
from utils.ollama_client import get_ollama_embedding

logger = logging.getLogger(__name__)


class ChromaStoreManager:
    """Manager for local persistent ChromaDB vector store and metadata tracking."""

    def __init__(
        self,
        persist_dir: str = CHROMA_PERSIST_DIR,
        collection_name: str = COLLECTION_NAME,
    ):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.chroma_client = chromadb.PersistentClient(path=self.persist_dir)
        self.chroma_collection = self.chroma_client.get_or_create_collection(self.collection_name)
        self.embed_model = get_ollama_embedding()

    def get_vector_store(self) -> ChromaVectorStore:
        return ChromaVectorStore(chroma_collection=self.chroma_collection)

    def get_index(self) -> VectorStoreIndex:
        vector_store = ChromaVectorStore(chroma_collection=self.chroma_collection)
        storage_context = StorageContext.from_defaults(vector_store=vector_store)
        return VectorStoreIndex.from_vector_store(
            vector_store=vector_store,
            storage_context=storage_context,
            embed_model=self.embed_model,
        )

    def is_document_indexed(self, doc_id: str) -> bool:
        """Check if a document is already indexed in the vector store."""
        try:
            results = self.chroma_collection.get(where={"document_id": doc_id}, limit=1)
            return len(results.get("ids", [])) > 0
        except Exception:
            return False

    def index_nodes(self, nodes: List[BaseNode]) -> int:
        """Index text nodes into ChromaDB. Skips already indexed chunks.

        Returns:
            Number of newly indexed nodes.
        """
        if not nodes:
            return 0

        # Filter out nodes whose document_id is already in collection
        new_nodes = []
        for node in nodes:
            doc_id = node.metadata.get("document_id")
            if doc_id and not self.is_document_indexed(doc_id):
                new_nodes.append(node)
            elif not doc_id:
                new_nodes.append(node)

        if not new_nodes:
            logger.info("All documents are already indexed in ChromaDB.")
            return 0

        index = self.get_index()
        index.insert_nodes(new_nodes)
        logger.info("Successfully indexed %d new chunks into ChromaDB.", len(new_nodes))
        return len(new_nodes)

    def retrieve(self, query: str, top_k: int = DEFAULT_TOP_K) -> List[Dict[str, Any]]:
        """Retrieve top-K context chunks for a query from ChromaDB.

        Returns:
            List of dicts containing text, score, and metadata.
        """
        index = self.get_index()
        retriever = VectorIndexRetriever(index=index, similarity_top_k=top_k)
        nodes_with_scores = retriever.retrieve(query)

        results = []
        for nws in nodes_with_scores:
            meta = nws.node.metadata
            results.append({
                "text": nws.node.get_content(),
                "score": float(nws.score or 0.0),
                "filename": meta.get("filename", "unknown.pdf"),
                "page_number": meta.get("page_number", 1),
                "chunk_id": meta.get("chunk_id", "c0"),
                "document_id": meta.get("document_id", "doc0"),
            })

        return results

    def get_statistics(self) -> Dict[str, int]:
        """Compute database statistics: Documents, Pages, Chunks, Embeddings."""
        count = self.chroma_collection.count()
        if count == 0:
            return {"documents": 0, "pages": 0, "chunks": 0, "embeddings": 0}

        # Fetch metadata to calculate unique documents and pages
        raw_data = self.chroma_collection.get(include=["metadatas"])
        metadatas = raw_data.get("metadatas", []) or []

        unique_docs = set()
        unique_pages = set()

        for meta in metadatas:
            if not meta:
                continue
            doc_id = meta.get("document_id") or meta.get("filename")
            page_num = meta.get("page_number")
            if doc_id:
                unique_docs.add(doc_id)
                if page_num is not None:
                    unique_pages.add(f"{doc_id}_p{page_num}")

        return {
            "documents": len(unique_docs),
            "pages": len(unique_pages),
            "chunks": count,
            "embeddings": count,
        }

    def clear_database(self) -> bool:
        """Clear all data from ChromaDB collection."""
        try:
            self.chroma_client.delete_collection(self.collection_name)
            self.chroma_collection = self.chroma_client.create_collection(self.collection_name)
            logger.info("ChromaDB collection '%s' cleared.", self.collection_name)
            return True
        except Exception as e:
            logger.error("Failed to clear ChromaDB: %s", str(e))
            return False
