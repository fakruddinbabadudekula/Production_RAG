import uuid

from fastapi import APIRouter, Depends
from app.models.user import User
from app.core.dependencies import get_current_user,get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.conversation_repository import conversation_repository
from app.schemas.history import SessionResponse,MessageResponse
from typing import List
router=APIRouter()

@router.get("/sessions",response_model=List[SessionResponse])
async def get_sessions(current_user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    return await conversation_repository.get_sessions(current_user.user_id,db)


        
@router.get("/messages",response_model=List[MessageResponse])
async def get_messages(session_id:uuid.UUID,current_user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    if session_id and await conversation_repository.verify_session(current_user.user_id,session_id,db):
        return await conversation_repository.get_messages(session_id,db)
