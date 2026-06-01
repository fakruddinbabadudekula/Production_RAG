import uuid

from fastapi import APIRouter, Depends
from app.models.user import User
from app.core.dependencies import get_current_user,get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.database import db_service
from app.schemas.history import SessionResponse,MessageResponse
from typing import List
router=APIRouter()

@router.get("/sessions",response_model=List[SessionResponse])
async def get_sessions(current_user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    return await db_service.get_sessions(current_user.user_id,db)


        
@router.get("/messages",response_model=List[MessageResponse])
async def get_messages(session_id:uuid.UUID,current_user:User=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    if session_id and await db_service.verify_session(session_id,current_user.user_id,db):
        return await db_service.get_messages(session_id,db)
