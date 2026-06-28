"""Rag Module for faiss vector store.
It is a wrapper aroud langchain FAISS vector store class to load and create faiss vector store.
"""

from functools import lru_cache
from pathlib import Path
import faiss
from langchain_community.vectorstores import FAISS
from langchain_core.embeddings import Embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore


class FaissStore:
    def load_vector_store(
        self,
        vector_dir_path: Path,
        embedding_client: Embeddings,
    ) -> FAISS:
        """Load existing vector Store

        Args:
                vector_dir_path: stores vector in that dir.
                embedding_client: converts the docs in embeddings.

        Returns:
                FAISS: Vector store."""

        vector_store = FAISS.load_local(
            vector_dir_path,
            embedding_client,
            allow_dangerous_deserialization=True,
        )

        return vector_store

    def create_vector_store(
        self, vector_dir_path: Path, embedding_client: Embeddings, embedding_len: int
    ) -> FAISS:
        """create a new vector store in a given vector_dir_path.
        Args:
            vector_dir_path: vector directory path where should vectors should be store.
            embedding_client: embedding client where it used to converts into embeddings.
            embedding_len: lenght of the embeddings.

        Returns:
            FAISS: Vector store

        Raises:
            ValueError: If embedding_len less that or equal to zero."""
        if embedding_len <= 0:
            raise ValueError("embedding_len must be greater that 0.")
        vector_dir_path.mkdir(parents=True, exist_ok=True)
        index = faiss.IndexFlatL2(embedding_len)
        vector_store = FAISS(
            embedding_function=embedding_client,
            index=index,
            docstore=InMemoryDocstore(),
            index_to_docstore_id={},
        )
        vector_store.save_local(vector_dir_path)

        return vector_store


@lru_cache(maxsize=1)
def get_module():
    return FaissStore()


faiss_store = get_module()
