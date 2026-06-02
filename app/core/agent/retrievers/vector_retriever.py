from langchain_community.vectorstores import FAISS
import faiss
import os
from pathlib import Path
from typing import List
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents.base import Document
import logging
from app.core.config import settings
from app.core.exceptions import InvalidFilePath, ProcessTimeOutError
from langchain_huggingface import HuggingFaceEmbeddings
from functools import lru_cache
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


@lru_cache()
def load_embeddings() -> HuggingFaceEmbeddings:
    """Load and cache the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: Cached embedding model instance.
        length: no of dimension are there.
    """
    # here All minilm models have the 384 dimension
    return HuggingFaceEmbeddings(model_name=settings.EMBED_MODEL)


def get_vector_path(user_id: str, session_id: str) -> Path:
    """Sanitize the file path
    raises:
        - InvalidFilePath: If any other paths are given
    """
    vector_dir_path = (settings.VECTOR_FOLDER / user_id / session_id).resolve()
    if not vector_dir_path.is_relative_to(settings.VECTOR_FOLDER):
        raise InvalidFilePath(
            "Invalid Vector Path", extra={"user_id": user_id, "session_id": session_id}
        )
    return vector_dir_path


class Retriever:
    """FAISS-based vector store retriever with persistence support.

    Initializes a FAISS vector database from disk if it exists,
    otherwise creates a new one and saves it locally.
    """

    def __init__(self, user_id: str, session_id: str):
        """Initialize the FAISS vector store and retriever.

        Args:
            vector_dir_path: Directory path for storing or loading the FAISS index.
        Raises:
            VectorStoreError: initialization failed.

        """
        self.user_id, self.session_id = user_id, session_id
        self.vector_dir_path = get_vector_path(user_id, session_id)
        self.embeddings = load_embeddings()
        self.embeddings_len = settings.EMBED_MODEL_SIZE
        try:
            self.vector_db = self._initialize_vector_db()
        except RETRYABLE_VECTOR_EXCEPTIONS as e:
            raise ProcessTimeOutError(
                "Retriever intialization Timeout after retries",
                extra={"user_id": self.user_id, "session_id": self.session_id},
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
                self.vector_dir_path,
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
                "Created new vectore store. vectorstore_path= %s", self.vector_dir_path
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
            ProcessTimeOutError: TimeOut Error
        Example:
         >>> await retriever.aadd_documents(docs)
        """
        if len(docs) == 0 or not docs:
            raise ValueError(f"Docs must be atleast one. Passed empty")
        try:
            ids = await self._aadd_documents_internal(docs)
            self.vector_db.save_local(self.vector_dir_path)
            logger.info(
                "Successfully added the %s docs into vectorestore_path= %s",
                len(docs),
                self.vector_dir_path,
            )
            return ids
        except RETRYABLE_VECTOR_EXCEPTIONS as e:
            raise ProcessTimeOutError(
                "Adding docs to vector store Timeout after retries",
                extra={
                    "count": settings.MAX_PDF_PROCESS_RETRY,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                },
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
            ProcessTimeOutError: TimeOut Error

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
            raise ProcessTimeOutError(
                "Retreiving Topk docs Timeout after retries",
                extra={
                    "count": settings.MAX_PDF_PROCESS_RETRY,
                    "user_id": self.user_id,
                    "session_id": self.session_id,
                },
            ) from e


@lru_cache()
def get_retriever(user_id: str, session_id: str):
    return Retriever(user_id, session_id)
