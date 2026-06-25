from functools import lru_cache
from pathlib import Path
from fastapi import UploadFile
from app.core.config import settings
from app.core.exceptions import (
    UnSupportedResource,
    ResourceNotFoundException,
    InvalidFilePaths,
    ValidationException,
)
from app.models.file import FileMetadata
from app.repositories.file_repository import file_repository
from app.rag.document_loaders.doc_loader import doc_loader
from app.services.vector_store_service import get_vector_service, get_vector_path
from app.repositories.file_metadata_repository import file_metadata_repository
import uuid
from typing import List
from logging import getLogger

logger = getLogger(__name__)


class FileIngestion:
    def __init__(self):
        pass

    def validate_file(self, file: UploadFile) -> str:
        """validate file
        Args:
            file:UploadFile = fastapi file object to read the content
        Returns:
            suffix:str = file type
        Raises:
            ResourceNotFoundException: If filename not found in the file object
            UnSUpportedResource: If if is unsupported by doc_loaders"""
        if not file.filename:
            ResourceNotFoundException("FileName_not_found")
        suffix = Path(file.filename).suffix
        if suffix not in doc_loader.get_supported_docs():
            # why we do here when also raise the same error in doc_loader, if we raise early no need to store file right, then why again do that in doc_loader bcz for independent handling without services doc_loader may handle the error.
            raise UnSupportedResource(
                "Unsupported file type",
                details={"file_name": file.filename, "type": suffix},
            )
        return suffix

    def get_file_id(self) -> uuid.UUID:
        """return uuid"""
        return uuid.uuid4()

    def get_file_path(
        self, user_id: str, session_id: str, file_id: str, file_type: str
    ) -> Path:
        """Sanitize the file path
        Args:
            user_id,session_id,file_id : str =
        Returns:
            file_path:Path = creates file root_folder/user_id/session_id/file_id.ext

        Raises:
            - InvalidFilePaths: If any other paths are given
        """

        """Here we put .pdf directly but later we can add muliple types"""
        file_path = Path(
            (f"{settings.FILE_UPLOAD_PATH / user_id / session_id/file_id}{file_type}")
        ).resolve()
        # Mainly this error handling doesn't need because user_id and session_id are given by us right, so no path traversals are done,but if we do manually passing the ids then we can do the sanitize the file paths
        if not file_path.is_relative_to(settings.FILE_UPLOAD_PATH):
            raise InvalidFilePaths(
                "invalid_file_path",
                details={
                    "user_id": str(user_id),
                    "session_id": str(session_id),
                    "file_id": str(file_id),
                },
            )
        return file_path

    async def ingest(self, file: UploadFile, user_id:uuid.UUID, session_id:uuid.UUID, db) -> FileMetadata:
        """Process docs ingestion from storing the raw file to embedding and storing the metadata in to database"""
        file_id = self.get_file_id()
        file_type = self.validate_file(file)
        file_path = self.get_file_path(
            str(user_id), str(session_id), str(file_id), file_type
        )
        try:
            file_size = await file_repository.save(file, file_path)
            try:

                docs = await doc_loader.process_document(file_path)
                vector_service=get_vector_service(get_vector_path(str(user_id), str(session_id)))
                doc_ids = await vector_service.aadd_documents(docs)
            except FileNotFoundError as e:
                raise ResourceNotFoundException(
                    str(e), details={"user_id": str(user_id), "session_id": str(session_id)}
                )
            except ValueError as e:
                raise ValidationException(
                    str(e), details={"user_id": user_id, "session_id": session_id}
                )
            file_object = await file_metadata_repository.insert(
                file_id, file.filename, "PDF", file_size, session_id, db
            )
        except Exception as e:
            # if any eception occurs please remove the uploaded file and vectors for now later we will analyze more effective solutions.
            self.cleanup_after_error(file_path,doc_ids,vector_service)
            raise
        return file_object

    async def cleanup_after_error(self,file_path:Path,doc_ids:List[str],vector_service):
        # db cleanup is already rollback so we only need to cleanup:
        # if file upload,vectorstore successfully but not got error while storing then we cleanup uploaded file and vectorestore(we can also do retries for database instead of cleaning but we implement further.)
        # if file uploaded but got error in vector store then clean uploaded files only.
        # to do this a simple approach is that takes the file_path, vector ids as arguments if both values are not none then we want to cleanup both values.if only filepath then only if both are none then it indicates storing the upload file raise the error.
        file_repository.delete(file_path)
        if not doc_ids==None:
                vector_service.adelete_documents(doc_ids)
            
            
            

@lru_cache()
def get_service():
    return FileIngestion()


file_ingestion = get_service()
