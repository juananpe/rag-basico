# -*- coding: utf-8 -*-
"""
bare_minimum.py — Ejemplo mínimo de RAG (Retrieval-Augmented Generation)

Pipeline:
  1. Cargar páginas web (URLs)
  2. Dividir en chunks
  3. Generar embeddings (vía OpenRouter)
  4. Guardar en ChromaDB
  5. Consultar por similitud semántica
  6. Generar respuesta final con LLM (vía OpenRouter)
"""

import os
import re
import chromadb
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openrouter_embedding import OpenRouterEmbedding


def cargar_urls(urls):
    """Descarga páginas web y extrae el texto."""
    docs = []
    for url in urls:
        resp = requests.get(url, timeout=15)
        soup = BeautifulSoup(resp.text, "lxml")
        texto = soup.get_text()
        docs.append(Document(page_content=texto, metadata={"source": url}))
    return docs


def generar_respuesta(api_key, pregunta, chunks, metadatas):
    """Envía la pregunta + contexto recuperado a un LLM vía OpenRouter."""
    # Construir el contexto incluyendo la fuente de cada chunk
    partes = []
    fuentes = {}  # número → URL real
    for i, (texto, meta) in enumerate(zip(chunks, metadatas), 1):
        fuente = meta.get("source", "desconocida")
        fuentes[i] = fuente
        partes.append(f"[Fuente {i}: {fuente}]\n{texto}")
    contexto = "\n\n---\n\n".join(partes)

    prompt = (
        "Eres un asistente útil. Responde a la pregunta del usuario "
        "basándote únicamente en el contexto proporcionado. "
        "Cita las fuentes que uses al final de tu respuesta (usa Referencia o Referencias si hay más de una: xxxx)."
        "En xxx no pongas Fuente N, sino la URL real. "
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
            "max_tokens": 900,
        },
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    respuesta = data["choices"][0]["message"]["content"]

    # Post-procesado: añadir las URLs reales al final
    urls_usadas = set()
    for num in sorted(fuentes):
        if f"[Fuente {num}]" in respuesta:
            urls_usadas.add(f"{num}: {fuentes[num]}")
    if urls_usadas:
        respuesta += "\n\n---\nFuentes citadas:\n" + "\n".join(
            f"  [{f}]" for f in sorted(urls_usadas)
        )

    return respuesta

# ── Configuración ──────────────────────────────────────────────────────────


load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

client = chromadb.PersistentClient(path="./chroma_db")
nombre = "donostia_avisos"
colecciones = [c.name for c in client.list_collections()]

# ── 1-4. Ingesta (solo si la colección no existe) ──────────────────────────

if nombre not in colecciones:
    print("Creando colección nueva...")

    # 1. Cargar documentos desde URLs
    urls = [
        "https://www.donostia.eus/es/avisos/afectaciones-trafico-gros-junio-6",
        "https://www.donostia.eus/es/avisos/san-bartolome-reunion-cidadania-8-junio",
        "https://www.donostia.eus/es/avisos/temporada-playas",
    ]
    print("  Cargando URLs...")
    docs = cargar_urls(urls)
    print(f"    → {len(docs)} documentos cargados")

    # 2. Dividir en chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = splitter.split_documents(docs)
    print(f"    → {len(chunks)} chunks creados")

    # Limpiar espacios extra del HTML
    def limpiar(texto):
        return re.sub(r"\s+", " ", texto).strip()

    # 3. Embedding + 4. Guardar en ChromaDB
    embed_fn = OpenRouterEmbedding(OPENROUTER_API_KEY)
    collection = client.create_collection(
        name=nombre, embedding_function=embed_fn)  # type: ignore
    collection.add(
        documents=[limpiar(c.page_content) for c in chunks],
        metadatas=[c.metadata for c in chunks],
        ids=[str(i) for i in range(len(chunks))],
    )
    print(
        f"    → Colección '{nombre}' creada con {collection.count()} vectores")

else:
    # La colección ya existe: cargarla sin reingerir
    embed_fn = OpenRouterEmbedding(OPENROUTER_API_KEY)
    collection = client.get_collection(
        name=nombre, embedding_function=embed_fn)  # type: ignore
    print(f"Colección '{nombre}' ya existe ({collection.count()} vectores). "
          f"Usando datos existentes.")

# ── 5. Consultar ───────────────────────────────────────────────────────────

pregunta = "¿Qué cortes de tráfico hay en Gros?"
resultados = collection.query(query_texts=[pregunta], n_results=3)

print(f"\n🔍 Pregunta: {pregunta}\n")
for i, (texto, meta) in enumerate(
    zip(resultados["documents"][0], resultados["metadatas"][0])  # type: ignore
):
    print(f"Resultado {i + 1}:")
    print(f"  Fuente : {meta.get('source', 'N/A')}")
    print(f"  Texto  : {texto[:200]}...")
    print()

# ── 6. Generar respuesta final con LLM ────────────────────────────────────

chunks_recuperados = resultados["documents"][0]    # type: ignore
metadatas_recuperados = resultados["metadatas"][0]  # type: ignore
print("🧠 Generando respuesta con LLM...\n")
respuesta = generar_respuesta(
    OPENROUTER_API_KEY, pregunta, chunks_recuperados, metadatas_recuperados
)
print(f"📝 Respuesta final:\n{respuesta}")
