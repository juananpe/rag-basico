# -*- coding: utf-8 -*-
"""
bare_minimum.py — Ejemplo mínimo de RAG (Retrieval-Augmented Generation)

Pipeline:
  1. Cargar páginas web (URLs)
  2. Dividir en chunks
  3. Generar embeddings (vía OpenRouter)
  4. Guardar en ChromaDB
  5. Consultar por similitud semántica
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
        name=nombre, embedding_function=embed_fn)
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
        name=nombre, embedding_function=embed_fn)
    print(f"Colección '{nombre}' ya existe ({collection.count()} vectores). "
          f"Usando datos existentes.")

# ── 5. Consultar ───────────────────────────────────────────────────────────

pregunta = "¿Qué cortes de tráfico hay en Gros?"
resultados = collection.query(query_texts=[pregunta], n_results=3)

print(f"\n🔍 Pregunta: {pregunta}\n")
for i, (texto, meta) in enumerate(
    zip(resultados["documents"][0], resultados["metadatas"][0])
):
    print(f"Resultado {i + 1}:")
    print(f"  Fuente : {meta.get('source', 'N/A')}")
    print(f"  Texto  : {texto[:200]}...")
    print()
