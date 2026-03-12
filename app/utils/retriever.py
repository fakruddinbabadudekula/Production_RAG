"""This file contains the vector_retriever utilities for the application."""
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
from app.core.config import settings

@lru_cache()
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: Cached embedding model instance.
        length: no of dimension are there.
    """
    # here All minilm models have the 384 dimension
    return HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)