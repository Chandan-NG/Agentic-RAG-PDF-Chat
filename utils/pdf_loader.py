"""PDF loading and parsing utilities using PyMuPDF (fitz) and LlamaIndex."""
import hashlib
import logging
from pathlib import Path
from typing import List, Union
import fitz  # PyMuPDF
from llama_index.core.schema import Document

logger = logging.getLogger(__name__)


def generate_document_id(filename: str, content: bytes) -> str:
    """Generate a unique document ID based on filename and content hash."""
    hasher = hashlib.md5()
    hasher.update(filename.encode("utf-8"))
    hasher.update(content)
    return hasher.hexdigest()[:16]


def load_pdf_document(file_path_or_bytes: Union[str, Path, bytes], filename: str) -> List[Document]:
    """Load a PDF document page by page using PyMuPDF and return LlamaIndex Document objects.

    Args:
        file_path_or_bytes: File path (str/Path) or raw byte buffer of the PDF.
        filename: Original filename of the PDF.

    Returns:
        List of LlamaIndex Document objects, each representing a page with rich metadata.
    """
    documents: List[Document] = []
    try:
        if isinstance(file_path_or_bytes, (str, Path)):
            doc_path = Path(file_path_or_bytes)
            with open(doc_path, "rb") as f:
                content_bytes = f.read()
            pdf_doc = fitz.open(doc_path)
        else:
            content_bytes = file_path_or_bytes
            pdf_doc = fitz.open(stream=content_bytes, filetype="pdf")

        doc_id = generate_document_id(filename, content_bytes)
        total_pages = len(pdf_doc)

        for page_num in range(total_pages):
            page = pdf_doc.load_page(page_num)
            page_text = page.get_text("text")

            if not page_text or not page_text.strip():
                continue

            metadata = {
                "filename": filename,
                "page_number": page_num + 1,
                "total_pages": total_pages,
                "document_id": doc_id,
            }

            llama_doc = Document(
                text=page_text,
                doc_id=f"{doc_id}_p{page_num + 1}",
                extra_info=metadata,
            )
            documents.append(llama_doc)

        pdf_doc.close()
        logger.info("Successfully loaded '%s' (%d pages, %d text pages)", filename, total_pages, len(documents))

    except Exception as e:
        logger.error("Failed to load PDF document '%s': %s", filename, str(e), exc_info=True)
        raise RuntimeError(f"Could not parse PDF '{filename}': {str(e)}") from e

    return documents
