"""Rag Module for vector_embeddings where it adds_documents and give_retriever to work with vector store."""

from functools import lru_cache
from langchain_community.vectorstores import VectorStore
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_core.documents.base import Document
from pathlib import Path
from typing import List
from logging import getLogger
import time

logger = getLogger(__name__)


class VectorStoreManager:
    """A class which proved add_documents and get_retriever."""

    async def aadd_documents(
        self, vector_store: VectorStore, vector_dir_path: Path, docs: List[Document]
    ) -> List[str]:
        """Add documents to a vector store and persist the updated index.

        Args:
            vector_store:
                The vector store that receives the documents.

            vector_dir_path:
                Directory where the updated vector store is saved.

            docs:
                Documents to embed and store.

        Returns:
            List[str]:
                IDs assigned to the stored documents.

        Raises:
            ValueError:
                If ``docs`` is empty.

        """

        start = time.perf_counter()
        if len(docs) == 0 or not docs:
            raise ValueError(f"Docs must be atleast one. Passed empty")
        docs_ids = await vector_store.aadd_documents(docs)
        vector_store.save_local(vector_dir_path)
        logger.info(
            "Successfully_added_docs_to_vector_store",
            extra={
                "count": len(docs),
                "vector_path": vector_dir_path,
                "duration": time.perf_counter() - start,
            },
        )
        return docs_ids

    async def adelete_documents(
        self, vector_store: VectorStore, doc_ids: List[str]
    ) -> bool | None:
        """Async delete by vector ID's and wrapper around vectore_store.adelete().

        Args:
            vectore_store: Vectore_store where we want to delete.
            doc_ids: List of IDs to delete. If `None`, delete all.
        Returns:
            `True` if deletion is successful, `False` otherwise, `None` if not
                implemented.
        """
        start = time.perf_counter()
        result = await vector_store.adelete(doc_ids)
        logger.info(
            "Succesfully_deleted_docs_from_vector_store",
            extra={"count": len(doc_ids), "duration": time.perf_counter() - start},
        )
        return result

    def get_retriever(
        self,
        vector_store: VectorStore,
        search_type: str = "similarity",
        search_kwargs: dict = {"k": 5},
    ) -> VectorStoreRetriever:
        """Returns vectore retriever,
        Args:
            vector_store: vectore_store to generate retreiver.
            search_type: default to 'similarity'
            search_kwargs: default to {'k':5} which returns the top_k docs.
        Returns:
            retriever: a retriever which used to get documents."""
        retriever = vector_store.as_retriever(
            search_type=search_type, search_kwargs=search_kwargs
        )

        return retriever


@lru_cache(maxsize=1)
def get_module() -> VectorStoreManager:
    return VectorStoreManager()


vector_store_manager = get_module()