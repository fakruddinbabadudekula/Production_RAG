"""Module for upload docs which process file validation,file storage,generate embeddings and store them in vector store.
contains upload rotuer"""

from typing import Optional
import uuid
from fastapi import APIRouter, Depends, Form, UploadFile, File
from app.core.db import get_db
from app.models.file import FileMetadata
from app.models.user import User
from app.api.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.file_ingestion import file_ingestion
from app.repositories.conversation_repository import conversation_repository

router = APIRouter()


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: Optional[uuid.UUID] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    """store the uploaded file in local storage and in vector storage"""
    user_id = current_user.user_id
    if session_id:
        await conversation_repository.verify_session(user_id, session_id, db)

    else:
        session = await conversation_repository.create_session(user_id, db)
        session_id = session.session_id

    file_metadata = await file_ingestion.ingest(file, user_id, session_id, db)

    return file_metadata
