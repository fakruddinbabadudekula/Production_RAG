"""Repository Module for file database for storing files metadata"""

from functools import lru_cache
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.transaction import transaction
from app.models.file import FileMetadata
from logging import getLogger

logger = getLogger(__name__)


class FileMetadataRepository:
    async def insert(
        self,
        file_id: uuid.UUID,
        name: str,
        type: str,
        size: int,
        session_id: uuid.UUID,
        db: AsyncSession,
    ) -> FileMetadata:
        new_file_metadata = FileMetadata(
            file_id=file_id,
            name=name,
            size=size,
            type=type,
            session_id=session_id,
        )
        # no need to retry why session_id is comming from trusted source
        async with transaction(db):
            db.add(new_file_metadata)
        await db.refresh(new_file_metadata)
        logger.info(
            "new file is created",
            extra={
                "file_id": str(file_id),
                "type": type,
                "size": size,
                "session_id": str(session_id),
            },
        )
        return new_file_metadata


@lru_cache()
def get_repository():
    return FileMetadataRepository()


file_metadata_repository = get_repository()
