"""
Shared pytest fixtures.

Sets fake API keys before any `src` module is imported (several modules
construct ChatOpenAI/OpenAIEmbeddings at import time) and provides a
FakeEmbeddings double so FAISS-backed tests never call the real OpenAI
embeddings endpoint.
"""

import os

os.environ.setdefault("OPENAI_API_KEY", "sk-fake-test-key")
os.environ.setdefault("TAVILY_API_KEY", "fake-tavily-key")

import numpy as np
import pytest


class FakeEmbeddings:
    """Drop-in stand-in for OpenAIEmbeddings with no network calls."""

    def embed_documents(self, texts):
        return [np.random.rand(8).tolist() for _ in texts]

    def embed_query(self, text):
        return np.random.rand(8).tolist()

    def __call__(self, text):
        return self.embed_query(text)


@pytest.fixture
def fake_embeddings(monkeypatch):
    """Patch src.rag.retriever_setup.embeddings with a fake, offline embedder."""
    import src.rag.retriever_setup as retriever_setup
    fake = FakeEmbeddings()
    monkeypatch.setattr(retriever_setup, "embeddings", fake)
    # Reset the module-level singleton so each test starts from a clean slate.
    monkeypatch.setattr(retriever_setup, "_faiss_vectorstore", None)
    return fake


@pytest.fixture(autouse=True)
def _isolate_description_file(tmp_path, monkeypatch):
    """
    Run every test from a temp working directory so tests never read or
    write the real description.txt in the project root.
    """
    monkeypatch.chdir(tmp_path)
