"""
Test script for RAG pipeline with ChromaDB + OpenRouter.

Accepts both PDF and TXT files.

Usage:
    python test_load.py sample.pdf
    python test_load.py sample.txt
    python test_load.py sample.pdf --query "¿Cuál es la misión del plan Gazteria?"
    python test_load.py es-bases.txt --collection bases_especificas_2026 --query "cuáles son los criterios de valoración"
"""

import argparse
import os
import re
from pathlib import Path

import chromadb
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openrouter_embedding import OpenRouterEmbedding
from pdf_loader import load_pdf


# ── Helpers ────────────────────────────────────────────────────────────────


def limpiar(texto: str) -> str:
    """Normaliza espacios en blanco."""
    return re.sub(r"\s+", " ", texto).strip()


def generar_respuesta(api_key: str, pregunta: str,
                      chunks: list[str], metadatas: list[dict]) -> str:
    """Envía la pregunta + contexto recuperado a un LLM vía OpenRouter."""
    partes = []
    fuentes: dict[int, str] = {}
    for i, (texto, meta) in enumerate(zip(chunks, metadatas), 1):
        fuente = meta.get("source", "desconocida")
        pagina = meta.get("page", "N/A")
        fuentes[i] = f"{fuente} (pág. {pagina})"
        partes.append(f"[Fuente {i}: {fuentes[i]}]\n{texto}")
    contexto = "\n\n---\n\n".join(partes)

    prompt = (
        "Eres un asistente útil. Responde a la pregunta del usuario "
        "basándote únicamente en el contexto proporcionado. "
        "Cita las fuentes que uses al final de tu respuesta "
        "(usa Referencia o Referencias si hay más de una: xxxx). "
        "En xxx no pongas Fuente N, sino la fuente real. "
        "Si el contexto no contiene información suficiente, indícalo claramente.\n\n"
        f"Contexto:\n{contexto}\n\n"
        f"Pregunta: {pregunta}\n\n"
        "Respuesta:"
    )

    resp = requests.post(
        "https://openrouter.ai/api/v1/chat/completions",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "deepseek/deepseek-v4-flash",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 2000,
        },
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    respuesta = data["choices"][0]["message"].get("content")
    # Some models return content=None with reasoning; fall back to reasoning
    if not respuesta:
        respuesta = data["choices"][0]["message"].get("reasoning", "")

    if not respuesta:
        return "El modelo no generó respuesta."

    # Post-procesado: añadir las fuentes reales al final
    urls_usadas = set()
    for num in sorted(fuentes):
        if f"[Fuente {num}]" in respuesta:
            urls_usadas.add(f"{num}: {fuentes[num]}")
    if urls_usadas:
        respuesta += "\n\n---\nFuentes citadas:\n" + "\n".join(
            f"  [{f}]" for f in sorted(urls_usadas)
        )

    return respuesta


def load_txt(txt_path: str, chunk_size: int = 800) -> list[Document]:
    """Lee un archivo .txt y devuelve una lista de Document (uno por bloque).

    Si el archivo es muy largo, se divide en bloques de chunk_size líneas
    para que cada Document tenga un tamaño razonable.
    """
    path = Path(txt_path)
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()

    # Si el archivo tiene pocas líneas, un solo Document; si no, bloques.
    if len(lines) <= chunk_size:
        return [Document(
            page_content=text,
            metadata={"source": path.name, "page": 0, "total_pages": 1},
        )]

    documents: list[Document] = []
    for i in range(0, len(lines), chunk_size):
        block = "\n".join(lines[i:i + chunk_size])
        documents.append(Document(
            page_content=block,
            metadata={
                "source": path.name,
                "page": i // chunk_size,
                "total_pages": (len(lines) + chunk_size - 1) // chunk_size,
            },
        ))
    return documents


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a PDF or TXT file into ChromaDB and query it with RAG."
    )
    parser.add_argument("file_path", help="Path to the PDF or TXT file")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of characters to print per page/block. Default: 500",
    )
    parser.add_argument(
        "--query",
        type=str,
        default="¿Cuál es la misión del plan Gazteria?",
        help="Question to ask the RAG system.",
    )
    parser.add_argument(
        "--force-reindex",
        action="store_true",
        help="Delete existing collection and re-ingest from scratch.",
    )
    parser.add_argument(
        "--n-results",
        type=int,
        default=5,
        help="Number of chunks to retrieve for the query. Default: 5",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default="openai/text-embedding-3-small",
        help="Embedding model to use. Default: openai/text-embedding-3-small",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default=None,
        help="ChromaDB collection name. Default: derived from filename.",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY no encontrada. "
            "Asegúrate de tenerla en un archivo .env."
        )

    # ── 0. Detectar tipo de archivo ────────────────────────────────────
    path = Path(args.file_path)
    ext = path.suffix.lower()
    if ext not in (".pdf", ".txt"):
        raise ValueError(f"Formato no soportado: '{ext}'. Usa .pdf o .txt")

    if args.collection is None:
        args.collection = path.stem.replace(" ", "_").lower()

    # ── 1. Cargar documento ────────────────────────────────────────────
    if ext == ".pdf":
        print(f"📄 Cargando PDF: {args.file_path}")
        documents = load_pdf(args.file_path)
        print(f"   → {len(documents)} páginas cargadas\n")
    else:
        print(f"📄 Cargando TXT: {args.file_path}")
        documents = load_txt(args.file_path)
        print(f"   → {len(documents)} bloques cargados\n")

    # Vista previa
    for index, document in enumerate(documents, start=1):
        print(f"--- Block {index} ---")
        print(document.page_content[: args.preview_chars])
        print(f"   Metadata: {document.metadata}")
        print()

    # ─ 2. Dividir en chunks ───────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    print(f"✂️  {len(chunks)} chunks creados (chunk_size=800, overlap=200)")

    # ── 3-4. Embedding + ChromaDB ──────────────────────────────────────
    client = chromadb.PersistentClient(path="./chroma_db")
    nombre = args.collection

    if args.force_reindex:
        try:
            client.delete_collection(name=nombre)
            print(f"️  Colección '{nombre}' eliminada para reindexado.")
        except Exception:
            pass

    colecciones = [c.name for c in client.list_collections()]
    embed_fn = OpenRouterEmbedding(api_key, model_name=args.embedding_model)

    if nombre not in colecciones:
        print(f"️  Creando colección '{nombre}' con embeddings...")
        collection = client.create_collection(
            name=nombre,
            embedding_function=embed_fn,  # type: ignore
        )
        collection.add(
            documents=[limpiar(c.page_content) for c in chunks],
            metadatas=[c.metadata for c in chunks],
            ids=[str(i) for i in range(len(chunks))],
        )
        print(f"   → Colección '{nombre}' creada con "
              f"{collection.count()} vectores")
    else:
        collection = client.get_collection(
            name=nombre,
            embedding_function=embed_fn,  # type: ignore
        )
        print(f"📚 Colección '{nombre}' ya existe "
              f"({collection.count()} vectores). Usando datos existentes.")

    # ─ 5. Consultar ───────────────────────────────────────────────────
    pregunta = args.query
    print(f"\n🔍 Consulta: {pregunta}\n")
    resultados = collection.query(
        query_texts=[pregunta], n_results=args.n_results)

    for i, (texto, meta) in enumerate(
        zip(resultados["documents"][0],
            resultados["metadatas"][0])  # type: ignore
    ):
        print(f"Resultado {i + 1}:")
        print(f"  Fuente : {meta.get('source', 'N/A')} "
              f"(pág. {meta.get('page', 'N/A')})")
        print(f"  Texto  : {texto[:250]}...")
        print()

    # ── 6. Generar respuesta final con LLM ─────────────────────────────
    chunks_recuperados = resultados["documents"][0]       # type: ignore
    metadatas_recuperados = resultados["metadatas"][0]    # type: ignore
    print("🧠 Generando respuesta con LLM (deepseek/deepseek-v4-flash)...\n")
    respuesta = generar_respuesta(
        api_key, pregunta, chunks_recuperados, metadatas_recuperados
    )
    print(f"📝 Respuesta final:\n{respuesta}")


if __name__ == "__main__":
    main()
