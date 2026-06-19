from functools import lru_cache

from fastapi import UploadFile
from pathlib import Path
import aiofiles
from logging import getLogger

logger = getLogger(__name__)


class FileRepository:
    def __init__(self):
        pass

    async def save(self, file: UploadFile, file_path: Path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = 0
        logger.info("file_writing_started")
        async with aiofiles.open(file_path, "wb") as out_file:

            while chunk := await file.read(1024 * 1024):  # Read in 1 MB chunks
                await out_file.write(chunk)
                file_size += len(chunk)
        logger.info(f"completed_file_writing with size of {file_size}")
        return file_size

    async def delete(self, file_path: Path):
        pass

@lru_cache()
def get_service():
    return FileRepository()

file_repository=get_service()