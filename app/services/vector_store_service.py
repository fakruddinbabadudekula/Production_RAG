"""Service module for vector_store where it add and delete docs and also return retriever"""

import time
from typing import List

from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores.base import VectorStoreRetriever
from app.rag.vector.faiss_vector_store import faiss_store
from app.rag.vector.vecor_store_manager import vector_store_manager
from functools import lru_cache
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from app.core.config import settings
import os
from app.core.exceptions import InvalidFilePaths
from pathlib import Path
from logging import getLogger

logger = getLogger(__name__)


@lru_cache()
def load_embeddings() -> GoogleGenerativeAIEmbeddings:
    """Load and cache the Google Generative AI embedding model.

    Returns:
        GoogleGenerativeAIEmbeddings: Cached embedding model instance.
        length: no of dimensions (768 for models/embedding-001).

    Raises:
        EnvironmentError: If GOOGLE_API_KEY is not configured.
    """
    if not getattr(settings, "GOOGLE_API_KEY", None) and not os.environ.get(
        "GOOGLE_API_KEY"
    ):
        raise EnvironmentError("GOOGLE_API_KEY is not set in environment or settings.")

    return GoogleGenerativeAIEmbeddings(
        model=settings.EMBED_MODEL,
        output_dimensionality=settings.EMBED_MODEL_SIZE,
        api_key=settings.GOOGLE_API_KEY,
    )


def get_vector_path(user_id: str, session_id: str) -> Path:
    """Sanitize the file path
    Raises:
        - NotFoundError: If any other paths are given
    """
    vector_dir_path = (settings.VECTOR_FOLDER / user_id / session_id).resolve()
    if not vector_dir_path.is_relative_to(settings.VECTOR_FOLDER):
        raise InvalidFilePaths(
            "Invalid Vector Path", details={"user_id": user_id, "session_id": session_id}
        )
    return vector_dir_path


class VectorStoreServiece:
    def __init__(self, vector_dir_path):
        self.embedding_client: Embeddings = load_embeddings()
        self.embedding_len: int = settings.EMBED_MODEL_SIZE
        self.vector_dir_path = vector_dir_path
        self.vector_store = self._initialize_vector_store()

    def _initialize_vector_store(self):
        start = time.perf_counter()
        if (self.vector_dir_path / "index.faiss").exists():
            vector_store = faiss_store.load_vector_store(
                self.vector_dir_path, self.embedding_client
            )
            logger.info(
                "Initialized existed vectore store successfully",
                extra={
                "vector_dir_path":self.vector_dir_path,
                "duration":time.perf_counter() - start,}
            )
            return vector_store
        else:
            vector_store = faiss_store.create_vector_store(
                self.vector_dir_path, self.embedding_client, self.embedding_len
            )
            logger.info(
                "Created new vectore store",
                extra={
                "vector_dir_path":self.vector_dir_path,
                "duration":time.perf_counter() - start,},
            )
            return vector_store

    async def aadd_documents(self, docs) -> List[str]:
        """wrapper around vectore_embedding.aadd_documents"""
        return await vector_store_manager.aadd_documents(
            self.vector_store, self.vector_dir_path, docs
        )
    async def adelete_documents(self,doc_ids:List[str]):
        """wrapper around vectore_embedding.adelete_documents"""
        return await vector_store_manager.adelete_documents(self.vector_store,doc_ids)

    def get_retriever(self)->VectorStoreRetriever:
        """later we can add serach_type and kwargs"""
        return vector_store_manager.get_retriever(self.vector_store)


@lru_cache()
def get_vector_service(vector_dir_path: Path):
    return VectorStoreServiece(vector_dir_path)
