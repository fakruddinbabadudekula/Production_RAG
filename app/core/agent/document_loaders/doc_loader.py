from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from app.core.exceptions import DocumentError
from typing import List
from pathlib import Path
import logging
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
    before_sleep_log,
)
import time
from functools import lru_cache

from app.core.config import settings

logger = logging.getLogger(__name__)

RETRYABLE_PDF_EXCEPTIONS = (
    TimeoutError,  # Network timeout (if loading from URL)
    ConnectionError,  # Network issues
    MemoryError,  # Temporary memory issues
    OSError,  # Temporary file system issues
)

# Define non-retryable errors (these are permanent failures)
PERMANENT_PDF_EXCEPTIONS = (
    FileNotFoundError,  # File doesn't exist - retry won't help
    PermissionError,  # No permission - retry won't help
    ValueError,  # Corrupt PDF - retry won't help
)


@lru_cache()
def get_recursive_splitter(
    chunk_size: int, chunk_overlap: int
) -> RecursiveCharacterTextSplitter:
    """Create and cache a configured RecursiveCharacterTextSplitter."""

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    return splitter


class DocumentLoader:

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        if chunk_size < 100:
            raise ValueError(f"chunk_size too small. value= {chunk_size} (min: 100)")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size.")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {".pdf"}

    def _validate_file(self, file_path: Path):
        """Validate the file
        Raise:
            - ValueError: If file type is not supported.
            - FileNotFoundError: If file is not found
        """
        if not file_path.resolve().is_relative_to(settings.DATA_PATH):
            raise ValueError(
                f" file_must_be_in_allowed_folder. path=> {file_path.name}"
            )
        if not file_path.exists():
            raise FileNotFoundError(f"file_not_found. file= {file_path}")
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported_file_format. file= {file_path.name} supported_formats= {self.supported_formats}")

    async def process_document(self, file_path: Path) -> List[Document]:
        """Process a document file and return list chunked LangChain Document objects.

        Args:
            file_path: File system path of the document to process.

        Returns:
            List[Document]: List of chunked Document objects.

        Raises:
            FileNotFoundError: If the file does not exist at the specified path.
            ValueError: If the file format is unsupported or processing fails.

        Example:
            >>> process_document(Path("data/sample.pdf"))
            [Document(...), Document(...)]
        """

        self._validate_file(file_path=file_path)
        logger.info("file_validated. file= %s",file_path.name)
        try:
            if file_path.suffix.lower() == ".pdf":
                docs = await self._process_pdf(file_path)
                return docs

        except RETRYABLE_PDF_EXCEPTIONS as e:
            raise DocumentError(
                message="document_process_after_retries_pdf_error",
                operation="pdf_processing",
                original_error=e,
                file_path=file_path.name
            ) from e
        except PERMANENT_PDF_EXCEPTIONS as e:
            raise DocumentError(
                message="document_process_permanant_error",
                operation="pdf_processing",
                original_error=e,
                file_path=file_path.name
            ) from e
        except Exception as e:
            raise DocumentError(
                message="document_process_unkown_error",
                operation="pdf_processing",
                original_error=e,
                file_path=file_path.name
            ) from e

    @retry(
        stop=stop_after_attempt(settings.MAX_PDF_PROCESS_RETRY), 
        wait=wait_exponential(multiplier=1, min=2, max=8),  # 2s, 4s, 8s
        retry=retry_if_exception_type(RETRYABLE_PDF_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process a PDF file and return chunked LangChain Document objects."""
        load_start = time.perf_counter()
        logger.info("started_pdf_process. file= %s", file_path.name)

        pdf_loader = PyMuPDFLoader(file_path=file_path)
        data = await pdf_loader.aload()
        load_duration=time.perf_counter()-load_start
        if not data:
            raise ValueError(f"contains_zero_pages. file= {file_path.name}")
        
        logger.info(
            "pdf_loaded file= %s pages= %s duration= %.3fs",
            file_path.name,
            len(data),
            load_duration
        )
        splitter = get_recursive_splitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks_start=time.perf_counter()
        docs = splitter.split_documents(data)
        chunks_duration=time.perf_counter()-chunks_start
        if not docs:
            raise ValueError(
                f"PDF loaded but produced no chunks. "
                f"File may be empty or contain only images: {file_path.name}"
            )

        logger.info("processed_pdf= file %s chunks= %s duration= %.3fs", file_path.name, len(docs),chunks_duration)
        return docs
