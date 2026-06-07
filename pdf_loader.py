"""
PDF loader using pypdf + langchain-core.
Load a PDF file and convert each page to a LangChain Document
"""

from pathlib import Path

import pypdf
from langchain_core.documents import Document


def load_pdf(pdf_path: str | Path) -> list[Document]:
    """
    Load a PDF file and convert each page to a LangChain Document.

    Args:
        pdf_path: Path to a PDF file.

    Returns:
        A list of LangChain Document objects, one per page.
    """
    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF file not found: {pdf_path}")

    if pdf_path.suffix.lower() != ".pdf":
        raise ValueError(f"Expected a .pdf file, got: {pdf_path}")

    reader = pypdf.PdfReader(str(pdf_path))

    documents: list[Document] = []

    for page_number, page in enumerate(reader.pages):
        text = page.extract_text() or ""

        documents.append(
            Document(
                page_content=text,
                metadata={
                    "source": str(pdf_path),
                    "page": page_number,
                    "total_pages": len(reader.pages),
                },
            )
        )

    return documents
