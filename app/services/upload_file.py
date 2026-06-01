from functools import lru_cache
import uuid
from pathlib import Path

from app.core.exceptions import UploadFileError
from fastapi import UploadFile
import os
import aiofiles
from app.core.config import settings
from logging import getLogger
logger=getLogger(__name__)
def get_file_path(user_id:str,session_id:str,file_id:str):
        """Sanitize the file path
        raises:
            - ValueError: If any other paths are given
        """
        file_path = Path((f"{settings.FILE_UPLOAD_PATH / user_id / session_id/file_id}.pdf")).resolve()
        if not file_path.is_relative_to(settings.FILE_UPLOAD_PATH):
            raise ValueError(
                f"Upload file address must be within the limit.Path=> {file_path}"
            )
        return file_path
class FileUpload:
    def get_file_id(self):        
        return uuid.uuid4()
    
    def validate_file(self, file: UploadFile):
        """For now i am only doing pdf validating only."""
        suffix = Path(file.filename).suffix
        if suffix == ".pdf":
            logger.info(f"successfully_validated_file and type {suffix}")
            return True
        error_msg= f"unsuported_file_type {suffix}"
        logger.error(error_msg)
        raise UploadFileError(
            error_msg,
            operation="validate_file",
            file_type=suffix,
            error_type="UnSupported_file_type"
        )

    async def store_file(
        self,
        file: UploadFile,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        file_id: uuid.UUID,
    ):
        file_path = get_file_path(user_id,session_id,file_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_size=0
        try:
            logger.info("file_writing_started")
            async with aiofiles.open(file_path, "wb") as out_file:
                
                while chunk := await file.read(1024 * 1024):  # Read in 1 MB chunks
                    await out_file.write(chunk)
                    file_size+=len(chunk)
            logger.info(f"completed_file_writing with size of {file_size}")
        except Exception as e:
            # Clean up partially written file in case of an error
            if os.path.exists(file_path):
                os.remove(file_path)
                logger.error(f"failed_to_write_file_data")
            raise

        return file_path,file_size


@lru_cache()
def get_upload_file_service():
    return FileUpload()


upload_file_service = get_upload_file_service()
