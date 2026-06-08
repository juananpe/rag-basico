# --- Simple text chunker (reemplaza langchain_text_splitters) ---

class SimpleTextSplitter:
    """Split text into overlapping chunks by character count."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 200, separator: str = "\n"):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separator = separator

    def split_documents(self, documents: list) -> list:
        chunks = []
        for doc in documents:
            text = doc["page_content"]
            # Split by separator first
            paragraphs = text.split(self.separator)
            current_chunk = ""
            for para in paragraphs:
                if len(current_chunk) + len(para) > self.chunk_size:
                    if current_chunk:
                        chunks.append({
                            "page_content": current_chunk.strip(),
                            "metadata": {**doc.get("metadata", {}), "chunk": len(chunks)},
                        })
                    # Keep overlap: take last chars from previous chunk
                    current_chunk = current_chunk[-self.chunk_overlap:] + para
                else:
                    current_chunk = (current_chunk + self.separator + para).strip() if current_chunk else para
            if current_chunk.strip():
                chunks.append({
                    "page_content": current_chunk.strip(),
                    "metadata": {**doc.get("metadata", {}), "chunk": len(chunks)},
                })
        return chunks
