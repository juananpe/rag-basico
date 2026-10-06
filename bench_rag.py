"""
Benchmark del retrieval del RAG: chunking × modelo de embeddings × rerank.

Para cada configuración indexa el PDF en memoria (similitud coseno, el mismo
ranking que ChromaDB con vectores normalizados) y mide, sobre un conjunto de
preguntas con su página correcta conocida:

  - hit@k : % de preguntas cuya página correcta aparece entre los k primeros chunks
  - MRR   : media de 1/posición del primer chunk correcto (1 = siempre el primero)
  - coste y latencia de indexado / consulta (según `usage.cost` de OpenRouter)

Uso:
    python bench_rag.py                       # fase 1: chunking × embeddings
    python bench_rag.py --rerank              # fase 2: + rerankers sobre las mejores
    python bench_rag.py --models openai/text-embedding-3-small --chunk 800:200
"""

import argparse
import os
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import requests
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from pdf_loader import load_pdf
from test_load_pdf import limpiar, quitar_boilerplate

PDF = "cas-plan-donostia-gazteria-2025-2027.pdf"
API = "https://openrouter.ai/api/v1"

# (pregunta, páginas correctas — numeradas desde 1 como en el visor de PDF)
PREGUNTAS = [
    ("¿Cuál es la misión del plan Gazteria?", {7}),
    ("¿Cuál es la visión del departamento de Juventud?", {7}),
    ("¿Qué valores tiene el departamento de Juventud?", {8}),
    ("¿Cuáles son los cuatro retos del plan?", {9}),
    ("¿Cuántas personas componen el equipo del departamento de Juventud?", {10}),
    ("¿Cuántas respuestas tuvo la encuesta dirigida a padres y madres?", {5}),
    ("¿Por qué se plantea como un plan de transición?", {4}),
    ("¿Cómo se evaluará el plan?", {20}),
    ("¿Qué rango de edad atiende el departamento de Juventud?", {3, 12}),
    ("¿Qué se hará con la Comisión Política de Juventud?", {18, 19}),
    ("¿Cuántas sesiones de reflexión se realizaron?", {6}),
    ("¿Cuándo se hace la valoración intermedia del plan de gestión anual?", {20}),
    ("¿Qué actividades culturales se organizan en el centro joven Kontadores?", {14}),
    ("¿Con qué redes o instituciones externas se colabora?", {16, 17}),
    ("¿Se va a abrir algún haurtxoko nuevo?", {15}),
]

EMBEDDING_MODELS = [
    "openai/text-embedding-3-small",
    "openai/text-embedding-3-large",
    "qwen/qwen3-embedding-8b",
    "qwen/qwen3-embedding-4b",
    "baai/bge-m3",
    "intfloat/multilingual-e5-large",
    "google/gemini-embedding-001",
    "voyageai/voyage-4-lite",
    "voyageai/voyage-4",
    "perplexity/pplx-embed-v1-0.6b",
    "mistralai/mistral-embed-2312",
]

RERANK_MODELS = [
    "cohere/rerank-4-fast",
    "cohere/rerank-v3.5",
    "voyageai/rerank-2.5-lite",
    "qwen/qwen3-reranker-8b",
]

def trocear(docs: list[Document], size: int, overlap: int,
            clean: bool) -> tuple[list[str], list[int]]:
    if clean:
        docs = [Document(page_content=quitar_boilerplate(d.page_content),
                         metadata=d.metadata) for d in docs]
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=size, chunk_overlap=overlap, length_function=len)
    chunks = [c for c in splitter.split_documents(docs)
              if limpiar(c.page_content)]
    return ([limpiar(c.page_content) for c in chunks],
            [c.metadata["page"] for c in chunks])


def post(api_key: str, path: str, payload: dict) -> dict:
    for intento in range(3):
        resp = requests.post(
            f"{API}/{path}",
            headers={"Authorization": f"Bearer {api_key}"},
            json=payload, timeout=120)
        if resp.status_code < 500 and resp.status_code != 429:
            break
        time.sleep(2 ** intento)
    resp.raise_for_status()
    data = resp.json()
    if "error" in data:
        raise RuntimeError(data["error"])
    return data


def embed(api_key: str, model: str, textos: list[str],
          batch: int = 64) -> tuple[np.ndarray, float]:
    vecs, coste = [], 0.0
    for i in range(0, len(textos), batch):
        data = post(api_key, "embeddings",
                    {"model": model, "input": textos[i:i + batch]})
        vecs += [d["embedding"]
                 for d in sorted(data["data"], key=lambda d: d["index"])]
        coste += data.get("usage", {}).get("cost", 0.0) or 0.0
    m = np.array(vecs, dtype=np.float32)
    return m / np.linalg.norm(m, axis=1, keepdims=True), coste


def rerank(api_key: str, model: str, query: str,
           docs: list[str]) -> tuple[list[int], float]:
    data = post(api_key, "rerank",
                {"model": model, "query": query, "documents": docs})
    orden = [r["index"] for r in
             sorted(data["results"], key=lambda r: -r["relevance_score"])]
    return orden, data.get("usage", {}).get("cost", 0.0) or 0.0


def metricas(rankings: list[list[int]]) -> dict:
    """rankings[i] = páginas de los chunks recuperados para la pregunta i."""
    res = {}
    for k in (1, 3, 5):
        res[f"hit@{k}"] = np.mean([bool(set(r[:k]) & gold)
                                   for r, (_, gold) in zip(rankings, PREGUNTAS)])
    rr = []
    for r, (_, gold) in zip(rankings, PREGUNTAS):
        pos = next((i for i, p in enumerate(r[:10], 1) if p in gold), None)
        rr.append(1 / pos if pos else 0.0)
    res["mrr"] = float(np.mean(rr))
    res["mision_pos"] = next(
        (i for i, p in enumerate(rankings[0][:20], 1) if p in PREGUNTAS[0][1]),
        None)
    return res


def evaluar(api_key, docs, model, size, overlap, clean, rerankers, n_cand):
    textos, paginas = trocear(docs, size, overlap, clean)
    t0 = time.time()
    m_docs, coste_idx = embed(api_key, model, textos)
    t_idx = time.time() - t0

    t0 = time.time()
    m_q, coste_q = embed(api_key, model, [q for q, _ in PREGUNTAS])
    t_q = (time.time() - t0) / len(PREGUNTAS)

    orden = np.argsort(-(m_q @ m_docs.T), axis=1)
    base = dict(model=model, chunk=f"{size}:{overlap}", clean=clean,
                n_chunks=len(textos), coste_idx=coste_idx,
                coste_q=coste_q / len(PREGUNTAS), t_idx=t_idx, t_q=t_q)
    filas = [{**base, "rerank": "-",
              **metricas([[paginas[j] for j in o] for o in orden])}]

    for rr_model in rerankers:
        rankings, coste_rr, t0 = [], 0.0, time.time()
        for (q, _), o in zip(PREGUNTAS, orden):
            cand = list(o[:n_cand])
            idx, c = rerank(api_key, rr_model, q, [textos[j] for j in cand])
            coste_rr += c
            rankings.append([paginas[cand[i]] for i in idx])
        filas.append({**base, "rerank": rr_model,
                      "coste_q": base["coste_q"] + coste_rr / len(PREGUNTAS),
                      "t_q": base["t_q"] + (time.time() - t0) / len(PREGUNTAS),
                      **metricas(rankings)})
    return filas


def imprimir(filas: list[dict]) -> None:
    filas = sorted(filas, key=lambda f: (-f["mrr"], -f["hit@3"], f["coste_q"]))
    print("\n| Modelo embeddings | Chunk | Limpio | Rerank | Chunks | hit@1 | "
          "hit@3 | hit@5 | MRR | Pos. misión | Coste indexado ($) | "
          "Coste/consulta ($) | Latencia/consulta (s) |")
    print("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for f in filas:
        print(f"| {f['model']} | {f['chunk']} | {'sí' if f['clean'] else 'no'} "
              f"| {f['rerank']} | {f['n_chunks']} | {f['hit@1']:.0%} | "
              f"{f['hit@3']:.0%} | {f['hit@5']:.0%} | {f['mrr']:.2f} | "
              f"{f['mision_pos'] or '>20'} | {f['coste_idx']:.6f} | "
              f"{f['coste_q']:.6f} | {f['t_q']:.2f} |")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    parser.add_argument("--models", nargs="+", default=EMBEDDING_MODELS)
    parser.add_argument("--chunk", nargs="+",
                        default=["800:200", "400:100", "1200:200"],
                        help="Pares size:overlap a probar")
    parser.add_argument("--clean", choices=["no", "si", "ambos"],
                        default="ambos", help="Quitar cabeceras/pies de página")
    parser.add_argument("--rerank", nargs="*", default=None,
                        help="Rerankers a probar (sin valor: todos)")
    parser.add_argument("--n-candidates", type=int, default=20,
                        help="Chunks recuperados por embeddings antes del rerank")
    args = parser.parse_args()

    load_dotenv()
    api_key = os.environ["OPENROUTER_API_KEY"]
    docs = load_pdf(PDF)
    rerankers = (RERANK_MODELS if args.rerank == [] else args.rerank) or []
    cleans = {"no": [False], "si": [True], "ambos": [False, True]}[args.clean]

    configs = [(m, *map(int, c.split(":")), cl)
               for m in args.models for c in args.chunk for cl in cleans]
    filas = []
    with ThreadPoolExecutor(max_workers=6) as pool:
        futuros = {pool.submit(evaluar, api_key, docs, m, s, o, cl,
                               rerankers, args.n_candidates): (m, s, o, cl)
                   for m, s, o, cl in configs}
        for fut, cfg in futuros.items():
            try:
                filas += fut.result()
            except Exception as e:  # un modelo caído no debe parar el resto
                print(f"⚠️  {cfg}: {e}")
    imprimir(filas)


if __name__ == "__main__":
    main()
