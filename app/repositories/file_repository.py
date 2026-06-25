from functools import lru_cache
import time
from fastapi import UploadFile
from pathlib import Path
import aiofiles
from logging import getLogger

logger = getLogger(__name__)


class FileRepository:
    def __init__(self):
        pass

    async def save(self, file: UploadFile, file_path: Path) -> int:
        """saves the file raw data into file system with the address of file_path"""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = 0
        start = time.perf_counter()
        async with aiofiles.open(file_path, "wb") as out_file:

            while chunk := await file.read(1024 * 1024):  # Read in 1 MB chunks
                await out_file.write(chunk)
                file_size += len(chunk)
        logger.info(
            "completed_file_writing",
            extra={
                "file_name": file.filename,
                "file_size": file_size,
                "duration": time.perf_counter() - start,
            },
        )
        return file_size

    async def delete(self, file_path: Path)->None:
        #Here we don't raise the exception if file_path is not there,bcz file_path is only at the time save method if that method raise the exception then we have the file so that we remove it, if not present there is no issues with that 
        start = time.perf_counter()
        if await aiofiles.os.path.exists(file_path):
            await aiofiles.os.remove(file_path)
            logger.info(
                "deleted file successfully",
                extra={"file_path": file_path, "duration": time.perf_counter() - start},
            )


@lru_cache()
def get_service():
    return FileRepository()


file_repository = get_service()
