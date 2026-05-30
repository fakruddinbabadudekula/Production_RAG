from functools import lru_cache
import uuid
from pathlib import Path

from app.schemas.enums import FileType
from fastapi import UploadFile
import os
import aiofiles
from app.core.config import settings

class FileUpload:
    def __init__(self):
        pass

    def get_file_id(self):
        
        return uuid.uuid4()
    def get_file_path(self,user_id,session_id,file_id):
        """Sanitize the file path
        raises:
            - ValueError: If any other paths are given
        """
        file_path = Path((f"{settings.FILE_UPLOAD_PATH / str(user_id) / str(session_id)/str(file_id)}.pdf")).resolve()
        if not file_path.is_relative_to(settings.FILE_UPLOAD_PATH):
            raise ValueError(
                f"Vector file address must be within the limit.Path=> {file_path}"
            )
        return file_path
    def validate_file(self, file: UploadFile):
        suffix = Path(file.filename).suffix
        if suffix == ".pdf":
            return
        raise ValueError("Unsupported file type")

    async def store_file(
        self,
        file: UploadFile,
        user_id: uuid.UUID,
        session_id: uuid.UUID,
        file_id: uuid.UUID,
    ):
        file_path = self.get_file_path(user_id,session_id,file_id)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        try:
            async with aiofiles.open(file_path, "wb") as out_file:
                while chunk := await file.read(1024 * 1024):  # Read in 1 MB chunks
                    await out_file.write(chunk)
        except Exception as e:
            # Clean up partially written file in case of an error
            if os.path.exists(file_path):
                os.remove(file_path)
            raise

        return file_path


@lru_cache()
def get_upload_file_service():
    return FileUpload()


upload_file_service = get_upload_file_service()
