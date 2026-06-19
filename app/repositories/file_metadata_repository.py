from functools import lru_cache
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.transaction import transaction
from app.models.file import FileMetadata


class FileMetadataRepository:
    def __init__(self):
        pass

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
        async with transaction(db):
            db.add(new_file_metadata)
        return new_file_metadata

@lru_cache()
def get_repository():
    return FileMetadataRepository()

file_metadata_repository=get_repository()