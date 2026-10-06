"""
Test script for pdf_loader.load_pdf — RAG pipeline with ChromaDB + OpenRouter.

Usage:
    python test_load_pdf.py sample.pdf
    python test_load_pdf.py sample.pdf --query "¿Cuál es la misión del plan Gazteria?"
    python test_load_pdf.py sample.pdf --interactive --preview-chars 0
    python test_load_pdf.py sample.pdf -i --rerank cohere/rerank-v3.5

Los valores por defecto (bge-m3, chunks de 1200/200, sin cabeceras ni pies de
página) salen de las pruebas de bench_rag.py; ver tabla en README.md.
"""

import argparse
import os
import re

import chromadb
import requests
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter

from openrouter_embedding import OpenRouterEmbedding
from pdf_loader import load_pdf


# ── Helpers ────────────────────────────────────────────────────────────────


# Cabeceras / pies de página que se repiten en todas las páginas del PDF.
# Si se dejan, todos los chunks se parecen a preguntas sobre "el plan Gazteria".
BOILERPLATE = [
    r"PLAN DONOSTIA GAZTERIA\s*\|\s*[A-ZÁÉÍÓÚÑ ]+",
    r"www\.\s*donostia\.eus/gazteria",
    r"Plan de la Sección de Juventud del Ayuntamiento de San Sebastián 2025-2027",
    r"KUDEAKETA_PLANA.*",
    r"Orrialdea \d+",
    r"(?m)^\s*\d{1,2}\s*$",  # números de página sueltos
]


def limpiar(texto: str) -> str:
    """Normaliza espacios en blanco."""
    return re.sub(r"\s+", " ", texto).strip()


def quitar_boilerplate(texto: str) -> str:
    """Elimina cabeceras, pies y números de página repetidos."""
    for patron in BOILERPLATE:
        texto = re.sub(patron, " ", texto)
    return texto


def rerank(api_key: str, model: str, pregunta: str,
           documentos: list[str]) -> list[tuple[int, float]]:
    """Reordena documentos por relevancia con un reranker de OpenRouter."""
    resp = requests.post(
        "https://openrouter.ai/api/v1/rerank",
        headers={"Authorization": f"Bearer {api_key}"},
        json={"model": model, "query": pregunta, "documents": documentos},
        timeout=60,
    )
    resp.raise_for_status()
    return [(r["index"], r["relevance_score"])
            for r in sorted(resp.json()["results"],
                            key=lambda r: -r["relevance_score"])]


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


def responder(collection, api_key: str, pregunta: str, n_results: int,
              rerank_model: str | None = None, n_candidates: int = 20) -> None:
    """Recupera los chunks más similares a la pregunta y genera la respuesta."""
    # ── 5. Consultar ───────────────────────────────────────────────────
    print(f"\n🔍 Consulta: {pregunta}\n")
    resultados = collection.query(
        query_texts=[pregunta],
        n_results=n_candidates if rerank_model else n_results)
    chunks_recuperados = resultados["documents"][0]       # type: ignore
    metadatas_recuperados = resultados["metadatas"][0]    # type: ignore
    puntuaciones = [f"Distancia: {d:.4f}"
                    for d in resultados["distances"][0]]  # type: ignore

    # ── 5b. Rerank (opcional): reordena los candidatos y se queda con n ──
    if rerank_model:
        print(f"🔀 Reordenando {len(chunks_recuperados)} candidatos "
              f"con {rerank_model}...\n")
        orden = rerank(api_key, rerank_model, pregunta,
                       chunks_recuperados)[:n_results]
        chunks_recuperados = [chunks_recuperados[i] for i, _ in orden]
        metadatas_recuperados = [metadatas_recuperados[i] for i, _ in orden]
        puntuaciones = [f"Relevancia: {s:.4f} (antes #{i + 1})"
                        for i, s in orden]

    for i, (texto, meta, punt) in enumerate(
        zip(chunks_recuperados, metadatas_recuperados, puntuaciones)
    ):
        print(f"Resultado {i + 1}:")
        print(f"  Fuente : {meta.get('source', 'N/A')} "
              f"(pág. {meta.get('page', 'N/A')})")
        print(f"  {punt}")
        print(f"  Texto  : {texto[:250]}...")
        print()

    # ── 6. Generar respuesta final con LLM ─────────────────────────────
    print("🧠 Generando respuesta con LLM (deepseek/deepseek-v4-flash)...\n")
    respuesta = generar_respuesta(
        api_key, pregunta, chunks_recuperados, metadatas_recuperados
    )
    print(f"📝 Respuesta final:\n{respuesta}")


# ── Main ───────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Load a PDF into ChromaDB and query it with RAG."
    )
    parser.add_argument("pdf_path", help="Path to the PDF file")
    parser.add_argument(
        "--preview-chars",
        type=int,
        default=500,
        help="Number of characters to print per page. Default: 500",
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
        default="baai/bge-m3",
        help="Embedding model to use. Default: baai/bge-m3",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=1200,
        help="Chunk size in characters. Default: 1200",
    )
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=200,
        help="Overlap between consecutive chunks. Default: 200",
    )
    parser.add_argument(
        "--keep-boilerplate",
        action="store_true",
        help="Do not strip repeated page headers/footers before chunking.",
    )
    parser.add_argument(
        "--rerank",
        type=str,
        default=None,
        metavar="MODEL",
        help="Rerank model (e.g. cohere/rerank-v3.5). Default: no rerank",
    )
    parser.add_argument(
        "--n-candidates",
        type=int,
        default=20,
        help="Chunks retrieved before reranking. Default: 20",
    )
    parser.add_argument(
        "--collection",
        type=str,
        default="plan_donostia_gazteria",
        help="ChromaDB collection name. Default: plan_donostia_gazteria",
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="Ask questions one after another (ignores --query).",
    )

    args = parser.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise RuntimeError(
            "OPENROUTER_API_KEY no encontrada. "
            "Asegúrate de tenerla en un archivo .env."
        )

    # ── 1. Cargar PDF ──────────────────────────────────────────────────
    print(f"📄 Cargando PDF: {args.pdf_path}")
    documents = load_pdf(args.pdf_path)
    print(f"   → {len(documents)} páginas cargadas\n")

    # Vista previa
    for index, document in enumerate(documents, start=1):
        print(f"--- Page {index} ---")
        print(document.page_content[: args.preview_chars])
        print(f"   Metadata: {document.metadata}")
        print()

    # ── 2. Dividir en chunks ───────────────────────────────────────────
    if not args.keep_boilerplate:
        for document in documents:
            document.page_content = quitar_boilerplate(document.page_content)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        length_function=len,
    )
    chunks = [c for c in splitter.split_documents(documents)
              if limpiar(c.page_content)]
    print(f"✂️  {len(chunks)} chunks creados (chunk_size={args.chunk_size}, "
          f"overlap={args.chunk_overlap})")

    # ── 3-4. Embedding + ChromaDB ──────────────────────────────────────
    client = chromadb.PersistentClient(path="./chroma_db")
    nombre = args.collection

    if args.force_reindex:
        try:
            client.delete_collection(name=nombre)
            print(f"🗑️  Colección '{nombre}' eliminada para reindexado.")
        except Exception:
            pass

    # Configuración con la que se indexa; si cambia, hay que reindexar
    config = {
        "pdf": args.pdf_path,
        "embedding_model": args.embedding_model,
        "chunk_size": args.chunk_size,
        "chunk_overlap": args.chunk_overlap,
        "boilerplate": args.keep_boilerplate,
    }
    colecciones = {c.name: c.metadata or {} for c in client.list_collections()}
    if nombre in colecciones and colecciones[nombre] != config:
        client.delete_collection(name=nombre)
        del colecciones[nombre]
        print(f"♻️  La configuración de '{nombre}' ha cambiado. Reindexando.")
    embed_fn = OpenRouterEmbedding(api_key, model_name=args.embedding_model)

    if nombre not in colecciones:
        print(f"🛠️  Creando colección '{nombre}' con embeddings...")
        collection = client.create_collection(
            name=nombre,
            embedding_function=embed_fn,  # type: ignore
            metadata=config,
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

    # ── 5-6. Consultar y generar respuesta ─────────────────────────────
    def preguntar(pregunta: str) -> None:
        responder(collection, api_key, pregunta, args.n_results,
                  args.rerank, args.n_candidates)

    if not args.interactive:
        preguntar(args.query)
        return

    print("\n💬 Modo interactivo. Escribe una pregunta "
          "(línea vacía, 'salir' o Ctrl+D para terminar).")
    while True:
        try:
            pregunta = input("\n❓ Pregunta: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not pregunta or pregunta.lower() in {"salir", "exit", "quit"}:
            break
        try:
            preguntar(pregunta)
        except requests.RequestException as e:
            print(f"⚠️  Error llamando a OpenRouter: {e}")

if __name__ == "__main__":
    main()
