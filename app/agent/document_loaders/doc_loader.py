from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
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
from functools import lru_cache

logging.basicConfig(level=logging.INFO)
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
            raise ValueError("chunk_size too small (min: 100)")
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be less than chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.supported_formats = {".pdf"}

    def _validate_file(self, file_path: Path):
        """Validate the file
        Raise:
            - ValueError: If file type is not supported.
            - FileNotFoundError: If file is not found
        """
        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {file_path}")
        if file_path.suffix.lower() not in self.supported_formats:
            raise ValueError(f"Unsupported file format: {file_path.suffix}")

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
        try:
            if file_path.suffix.lower() == ".pdf":
                return await self._process_pdf(file_path)

        except RETRYABLE_PDF_EXCEPTIONS as e:
            logger.error(
                f"Error while processing after retries. {file_path.name}:{type(e).__name__}: {str(e)}"
            )
            raise

    @retry(
        stop=stop_after_attempt(3),  # Try 3 times max
        wait=wait_exponential(multiplier=1, min=2, max=8),  # 2s, 4s, 8s
        retry=retry_if_exception_type(RETRYABLE_PDF_EXCEPTIONS),
        before_sleep=before_sleep_log(logger, logging.WARNING),
        reraise=True,  # Raise the original exception after all retries fail
    )
    async def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process a PDF file and return chunked LangChain Document objects."""

        logger.info(f"Started Processing document: {file_path.name}")

        pdf_loader = PyMuPDFLoader(file_path=file_path)
        data = await pdf_loader.aload()
        if not data:
            raise ValueError(f"PDF contains no pages: {file_path.name}")
        splitter = get_recursive_splitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        docs = splitter.split_documents(data)
        if not docs:
            raise ValueError(
                f"PDF loaded but produced no chunks. "
                f"File may be empty or contain only images: {file_path.name}"
            )

        logger.info(f"Processed document: {file_path.name}")
        return docs
