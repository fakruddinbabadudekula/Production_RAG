"""Repository Module for storing raw files in filesystem"""
from functools import lru_cache
import time
from fastapi import UploadFile
from pathlib import Path
import aiofiles, aiofiles.os
from logging import getLogger

logger = getLogger(__name__)

CHUNK_SIZE = 1024 * 1024

class FileRepository:

    async def save(self, file: UploadFile, file_path: Path) -> int:
        """Save an uploaded file to the local file system.

        Args:
            file: UploadFile
                Uploaded file.

            file_path: Path
                Destination path where the file will be stored.

        Returns:
            int:
                Size of the saved file in bytes.

        Raises:
            OSError:
                If writing the file fails.
        """
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_size = 0
        start = time.perf_counter()
        async with aiofiles.open(file_path, "wb") as out_file:

            while chunk := await file.read(CHUNK_SIZE):  # Read in 1 MB chunks
                await out_file.write(chunk)
                file_size += len(chunk)
        logger.info(
            "completed file writing",
            extra={
                "file_name": file.filename,
                "file_size": file_size,
                "duration": time.perf_counter() - start,
            },
        )
        await file.close()
        return file_size

    async def delete(self, file_path: Path) -> None:
        """Delete a file from the local file system.

        Args:
            file_path: Path
                Path of the file to delete.
        """
        # Here we don't raise the exception if file_path is not there,becuase file_path is created in upper layer so there is trust,but it have two option weather the data is created or not, there is no issue with this os remove entire file.
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
