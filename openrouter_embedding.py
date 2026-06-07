# -*- coding: utf-8 -*-
"""
Adaptador ChromaDB para usar embeddings de OpenRouter (OpenAI-compatible).
"""

from openai import OpenAI


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
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self.model_name = model_name

    def name(self):
        return "openrouter-text-embedding-3-small"

    def _embed(self, input):
        resp = self.client.embeddings.create(
            model=self.model_name,
            input=input,
        )
        sorted_data = sorted(resp.data, key=lambda x: x.index)
        return [item.embedding for item in sorted_data]

    def __call__(self, input):
        """Usado por ChromaDB en collection.add()."""
        return self._embed(input)

    def embed_query(self, input):
        """Usado por ChromaDB en collection.query()."""
        return self._embed(input)
