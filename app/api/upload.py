from gc import get_referents
from typing import Optional
import uuid

from pathlib import Path

from sklearn.cluster import dbscan


from app.services.database import db_service
from fastapi import APIRouter, Depends, Form, UploadFile, File
from app.core.db import get_db
from app.models.user import User
from app.core.dependencies import get_current_user
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.upload_file import upload_file_service
from app.core.agent.document_loaders.doc_loader import doc_loader
from app.core.agent.retrievers.vector_retriever import get_retriever, get_vector_path
from app.services.database import db_service

router = APIRouter()


@router.post("/upload")
async def upload(
    file: UploadFile = File(...),
    session_id: Optional[uuid.UUID] = Form(None),
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
):
    # validate session id and validate file and store it and do embeddings
    user_id = current_user.user_id
    if session_id:
        await db_service.verify_session(session_id, user_id, db)

    else:
        session = await db_service.create_session(user_id, db)
        session_id = session.session_id

    file_id = upload_file_service.get_file_id()

    # validate and store the file in file system=> return metadata of the file like size and type
    upload_file_service.validate_file(file)
    file_path = await upload_file_service.store_file(file, user_id, session_id, file_id)

    # embedd the file and store it in vector store
    docs = await doc_loader.process_document(file_path)
    vector_path = get_vector_path(user_id, session_id)
    retriever = get_retriever(vector_path)
    ids = await retriever.aadd_documents(docs)
    # store the metadata in database
    file_metadata = await db_service.add_file(
        file_id, file.filename, "pdf", session_id, db
    )
    # return the session_id,file_metadata

    return file_metadata
