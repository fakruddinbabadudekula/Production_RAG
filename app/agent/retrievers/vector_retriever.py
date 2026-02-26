from langchain_community.vectorstores import FAISS
import faiss
import os
from pathlib import Path
from typing import List, Tuple
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_core.documents.base import Document
import logging
from functools import lru_cache
from app.config import settings
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
EMBED_MODEL = settings.EMBED_MODEL
EMBED_MODEL_SIZE = 384

RETRYABLE_VECTOR_EXCEPTIONS = (
    ConnectionError,  # Network issues (remote embeddings)
    TimeoutError,  # API/network timeouts
    MemoryError,  # Temporary memory pressure
    OSError,  # File system issues (FAISS index I/O)
    RuntimeError,  # FAISS internal errors (sometimes transient)
)


class VectorStoreError(Exception):
    """Custom exception for vector store operations."""

    def __init__(
        self, message: str, file_path: Path, operation: str, original_error: Exception
    ):
        full_message = f"{message}. Original error: {type(original_error).__name__}: {str(original_error)}"
        logger.error(
            f"message: {full_message} | operation: {operation} exception: {str(original_error)}| file_path: {file_path}"
        )
        super().__init__(full_message)
        self.operation = operation
        self.original_error = original_error
        self.file_path = file_path


@lru_cache()
def load_embeddings() -> Tuple[HuggingFaceEmbeddings, int]:
    """Load and cache the HuggingFace embedding model.

    Returns:
        HuggingFaceEmbeddings: Cached embedding model instance.
        length: no of dimension are there.
    """
    # here All minilm models have the 384 dimension
    return (
        HuggingFaceEmbeddings(model_name=EMBED_MODEL),
        EMBED_MODEL_SIZE,
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

        """

        self.vector_dir_path = vector_dir_path
        self.embeddings, self.embeddings_len = load_embeddings()
        try:
            self.vector_db = self._initialize_vector_db()
        except RETRYABLE_VECTOR_EXCEPTIONS as v_e:
            raise VectorStoreError(
                f"Failed to initialize vector store after retries",
                operation="Initialization",
                file_path=self.vector_dir_path,
                original_error=v_e,
            )
        except Exception as e:
            logging.error(
                f"Unknown Error Occured while initializing the vectorDB for path: {self.vector_dir_path}"
            )
            raise
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
            logger.info(f"Existing vectorstore found {self.vector_dir_path.name}")
            vector_db = FAISS.load_local(
                self.vector_dir_path,
                self.embeddings,
                allow_dangerous_deserialization=True,
            )

        else:
            logger.info(f"Creating vector dir {self.vector_dir_path}")
            os.makedirs(self.vector_dir_path, exist_ok=True)
            self.index = faiss.IndexFlatL2(self.embeddings_len)
            vector_db = FAISS(
                embedding_function=self.embeddings,
                index=self.index,
                docstore=InMemoryDocstore(),
                index_to_docstore_id={},
            )
            vector_db.save_local(self.vector_dir_path)
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
        self.vector_db.save_local(self.vector_dir_path)
        logger.info(f"Successfully added the docs into {self.vector_dir_path}")
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
            logger.error(f"Docs must be atleast one.")
            raise ValueError(f"Docs must be atleast one.Passed empty")
        try:
            return await self._aadd_documents_internal(docs)
        except RETRYABLE_VECTOR_EXCEPTIONS as v_e:
            raise VectorStoreError(
                f"Failed to add documents into vector store after retries",
                operation="AddingDocuments",
                file_path=self.vector_dir_path,
                original_error=v_e,
            )
        except Exception as e:
            logger.error(
                f"Adding documents throws an Unknown error {str(e)} at {self.vector_dir_path}"
            )
            raise
    
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
        logger.info(f"Successfully Perform the Top-K Retriever {query}")
        return top_docs

    async def aget_top_k(self, query: str) -> List[Document] | None:
        """Asynchronously retrieve the top-k most similar documents.

        Args:
            query: Search query string.

        Returns:
            List[Document]: List of top-k similar documents.

        Raises:
            VectorStoreError: If retrieval fails.
            ValueError: If docs are empty.

        Example:
            >>> docs = await retriever.aget_top_k("What is RAG?")
        """
        if not query or not query.strip():
            logger.error(f"query should not be empty.")
            raise ValueError("query cannot be empty")

        try:
            return await self._aget_top_k_internal(query)
        except RETRYABLE_VECTOR_EXCEPTIONS as v_e:
            raise VectorStoreError(
                f"Failed to retrive top_k docs from the vector store after retries",
                operation="RetrievingDocuments",
                file_path=self.vector_dir_path,
                original_error=v_e,
            )
        except Exception as e:
            logger.error(f"UnknownError at retrieving the docs {str(e)}")
            raise
