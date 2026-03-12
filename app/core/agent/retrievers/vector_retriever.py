from langchain_community.vectorstores import FAISS
import faiss
import os
from pathlib import Path
from typing import List
from app.utils.retriever import load_embeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents.base import Document
import logging
from app.core.config import settings
from app.core.exceptions import VectorStoreError
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logger = logging.getLogger(__name__)

RETRYABLE_VECTOR_EXCEPTIONS = (
    ConnectionError,  # Network issues (remote embeddings)
    TimeoutError,  # API/network timeouts
    MemoryError,  # Temporary memory pressure
    OSError,  # File system issues (FAISS index I/O)
    RuntimeError,  # FAISS internal errors (sometimes transient)
)


class Retriever:
    """FAISS-based vector store retriever with persistence support.

    Initializes a FAISS vector database from disk if it exists,
    otherwise creates a new one and saves it locally.
    """

    def __init__(self, vector_dir_path: Path):
        """Initialize the FAISS vector store and retriever.

        Args:
            vector_dir_path: Directory path for storing or loading the FAISS index.
        Raises:
            VectorStoreError: initialization failed.

        """

        self.vector_dir_path = vector_dir_path
        self.embeddings = load_embeddings()
        self.embeddings_len = settings.EMBED_MODEL_SIZE
        try:
            self.vector_db = self._initialize_vector_db()
        except RETRYABLE_VECTOR_EXCEPTIONS as e:
            raise VectorStoreError(
                f"initialize vector store after retries",
                operation="initialization",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f" Unkown error raised at the time of initialization",
                operation="initialization",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e
        self.retriever = self.vector_db.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        )

    @retry(
        stop=stop_after_attempt(3),  # Try 3 times max
        wait=wait_exponential(multiplier=1, min=2, max=8),  # 2s, 4s, 8s
        retry=retry_if_exception_type(RETRYABLE_VECTOR_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    def _initialize_vector_db(self):
        """Initialize the Vector_database,return already existed if not creates one."""
        if (self.vector_dir_path / "index.faiss").exists():
            logger.info(
                "Initializing existing vectorstore. vectorstore= %s",
                self.vector_dir_path.name,
            )
            vector_db = FAISS.load_local(
                self.vector_dir_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )
            logger.info(
                "Initialized existed vectore store successfully. vectorstore= %s",
                self.vector_dir_path.name,
            )

        else:
            logger.info(
                "Creating new vectore store. vectorstore= %s", self.vector_dir_path
            )
            os.makedirs(self.vector_dir_path, exist_ok=True)
            index = faiss.IndexFlatL2(self.embeddings_len)
            vector_db = FAISS(
                embedding_function=self.embeddings,
                index=index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            vector_db.save_local(self.vector_dir_path)
            logger.info(
                "Created new vectore store. vectorstore= %s", self.vector_dir_path
            )

        return vector_db

    @retry(
        stop=stop_after_attempt(5),  # Try 3 times max bcz it is imported
        wait=wait_exponential(multiplier=1, min=2, max=32),  # 2s, 4s, 8s,16s,32s
        retry=retry_if_exception_type(RETRYABLE_VECTOR_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _aadd_documents_internal(self, docs: List[Document]) -> list[str]:
        """Internal method to add documents
        return:
            docs_ids=id's of the added docs in vectors
        """
        docs_ids = await self.vector_db.aadd_documents(docs)
        return docs_ids

    async def aadd_documents(self, docs: List[Document]) -> list[str]:
        """Asynchronously add documents to the vector store.

        Args:
            docs: List of LangChain Document objects to embed and store.

        Raises:
            VectorStoreError: If adding documents to the vector store fails.
            ValueError: if docs are empty
        Example:
         >>> await retriever.aadd_documents(docs)
        """
        if len(docs) == 0 or not docs:
            raise ValueError(f"Docs must be atleast one.Passed empty")
        try:
            ids = await self._aadd_documents_internal(docs)
            self.vector_db.save_local(self.vector_dir_path)
            logger.info(
                "Successfully added the %s docs into vectorestore= %s",
                len(docs),
                self.vector_dir_path.name,
            )
            return ids
        except RETRYABLE_VECTOR_EXCEPTIONS as e:
            raise VectorStoreError(
                f"add_documents after retries",
                operation="adding_docs",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f" Unkown_error_adding_docs",
                operation="adding_docs",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e

    @retry(
        stop=stop_after_attempt(3),  # Try 3 times
        wait=wait_exponential(multiplier=1, min=2, max=8),  # 2s, 4s, 8s
        retry=retry_if_exception_type(RETRYABLE_VECTOR_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _aget_top_k_internal(self, query: str) -> List[Document] | None:
        """Internal method to get_top_k docs"""
        top_docs = await self.retriever.ainvoke(query)
        return top_docs

    async def aget_top_k(self, query: str) -> List[Document] | None:
        """Asynchronously retrieve the top-k most similar documents.

        Args:
            query: Search query string.

        Returns:
            List[Document] | None: List of top-k similar documents.

        Raises:
            VectorStoreError: If retrieval fails.
            ValueError: If docs are empty.

        Example:
            >>> docs = await retriever.aget_top_k("What is RAG?")
        """
        if not query or not query.strip():
            raise ValueError("query cannot be empty")

        try:
            top_k = await self._aget_top_k_internal(query)
            logger.info(
                "Successfully Perform the Retriever. Got %s docs for the query %s....",
                len(top_k),
                query[:10],
            )
            return top_k
        except RETRYABLE_VECTOR_EXCEPTIONS as e:
            raise VectorStoreError(
                f"retrieving_top_docs after retries",
                operation="retrieving_top_docs",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e
        except Exception as e:
            raise VectorStoreError(
                f" Unkown_error_retrieving_top_docs",
                operation="retrieving_top_docs",
                file_path=self.vector_dir_path,
                original_error=e,
            ) from e
