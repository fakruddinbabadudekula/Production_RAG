from functools import lru_cache
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings
from app.models.file import FileMetadata
from app.repositories.file_repository import file_repository
from app.rag.document_loaders.doc_loader import doc_loader
from app.services.vector_store_service import get_vector_service, get_vector_path
from app.repositories.file_metadata_repository import file_metadata_repository
import uuid
from logging import getLogger

logger = getLogger(__name__)


class FileIngestion:
    def __init__(self):
        pass

    def validate_file(self, file: UploadFile):
        if not file.filename:
            ValueError("FileName_not_found")
        suffix = Path(file.filename).suffix
        if not suffix == ".pdf":
            logger.warning("file_not_supported file_type = %s",suffix)
            raise ValueError("file_not_supported")
        return suffix

    def get_file_id(self):
        return uuid.uuid4()

    def get_file_path(self, user_id: str, session_id: str, file_id: str):
        """Sanitize the file path
        raises:
            - ValueError: If any other paths are given
        """

        """Here we put .pdf directly but later we can add muliple types"""
        file_path = Path(
            (f"{settings.FILE_UPLOAD_PATH / user_id / session_id/file_id}.pdf")
        ).resolve()
        if not file_path.is_relative_to(settings.FILE_UPLOAD_PATH):
            raise ValueError(
                "invalid_file_path",
            )
        return file_path

    async def ingest(self, file: UploadFile, user_id, session_id, db)->FileMetadata:
        file_id = self.get_file_id()
        file_type = self.validate_file(file)
        file_path = self.get_file_path(str(user_id), str(session_id), str(file_id))
        file_size = await file_repository.save(file, file_path)
        docs = await doc_loader.process_document(file_path)
        _ = await get_vector_service(
            get_vector_path(str(user_id), str(session_id))
        ).aadd_documents(docs)
        file_object = await file_metadata_repository.insert(
            file_id, file.filename, "PDF", file_size, session_id, db
        )
        return file_object

@lru_cache()
def get_service():
    return FileIngestion()

file_ingestion=get_service()