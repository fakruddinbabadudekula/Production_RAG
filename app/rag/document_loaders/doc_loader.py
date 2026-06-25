from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents.base import Document
from typing import List
from pathlib import Path
import logging
import time
from functools import lru_cache

logger = logging.getLogger(__name__)


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
        """Validates given file if exist or not and also check supported file_types
        Raise:
            - ValueError: If file type is not supported.
            - FileNotFoundError: If file is not found
        """
        if not file_path.exists():
            raise FileNotFoundError(f"file_not_found. path = {file_path}")
        file_type = file_path.suffix.lower()
        if file_type not in self.supported_formats:
            raise ValueError(
                f"Invalid_File_type or unsupported_file_type, type = {file_type}"
            )

    async def process_document(self, file_path: Path) -> List[Document]:
        """Process a document file and return list of chunked LangChain Document objects.

        Args:
            file_path:Path = File system path of the document to process.

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

        file_type = file_path.suffix.lower()
        if file_type == ".pdf":
            docs = await self._process_pdf(file_path)
            return docs

    async def _process_pdf(self, file_path: Path) -> List[Document]:
        """Process a PDF file and return chunked LangChain Document objects."""
        load_start = time.perf_counter()
        pdf_loader = PyMuPDFLoader(file_path=file_path)
        data = await pdf_loader.aload()
        load_duration = time.perf_counter() - load_start
        logger.info(
            "pdf_loaded",
            extra={
                "file_path": file_path.name,
                "page": len(data),
                "duration": load_duration,
            },
        )
        splitter = get_recursive_splitter(
            chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap
        )
        chunks_start = time.perf_counter()
        docs = splitter.split_documents(data)
        logger.info(
            "processed_pdf",
            extra={
                "chunk_size": self.chunk_size,
                "chunk_overlap": self.chunk_overlap,
                "file_name": file_path.name,
                "chunks": len(docs),
                "duration": time.perf_counter() - chunks_start,
            },
        )
        return docs
    def get_supported_docs(self):
        return self.supported_formats


@lru_cache()
def get_doc_loader():
    return DocumentLoader()


doc_loader = get_doc_loader()
