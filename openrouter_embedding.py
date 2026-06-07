# -*- coding: utf-8 -*-
"""
Adaptador ChromaDB para usar embeddings de OpenRouter sin depender del SDK de OpenAI.
Usa requests HTTP directamente para evitar incompatibilidades del SDK de OpenAI.
"""

import requests


class OpenRouterEmbedding:
    """
    Embedding function compatible con ChromaDB que usa el API de OpenRouter.

    Uso:
        embed_fn = OpenRouterEmbedding(api_key="sk-or-v1-...")
        collection = client.create_collection(
            name="mi_coleccion",
            embedding_function=embed_fn,
        )
    """

    def __init__(self, api_key, model_name="openai/text-embedding-3-small"):
        self.api_key = api_key
        self.model_name = model_name

    def name(self):
        # ChromaDB fallback: return a stable name so ChromaDB can cache dimensions
        safe = self.model_name.replace("/", "_").replace("-", "_")
        return f"openrouter_{safe}"

    def _embed(self, input):
        resp = requests.post(
            "https://openrouter.ai/api/v1/embeddings",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": input,
            },
            timeout=120,
        )
        resp.raise_for_status()
        data = resp.json()
        sorted_data = sorted(data["data"], key=lambda x: x["index"])
        return [item["embedding"] for item in sorted_data]

    def __call__(self, input):
        """Usado por ChromaDB en collection.add()."""
        return self._embed(input)

    def embed_query(self, input):
        """Usado por ChromaDB en collection.query()."""
        return self._embed(input)
