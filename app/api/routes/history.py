"""Module for get the history of user related data like sessions and messages.
need to change the name instead of history.
contains sessions and messages which returns the list of sessions or messages"""

import uuid
from fastapi import APIRouter, Depends
from app.models.user import User
from app.api.dependencies import get_current_user, get_db
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.conversation_repository import conversation_repository
from app.api.schemas.history import SessionResponse, MessageResponse
from app.models.session import Session
from app.models.message import Message
from typing import List

router = APIRouter()


@router.get("/sessions", response_model=List[SessionResponse])
async def get_sessions(
    current_user: User = Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> List[Session]:
    """get the list of conversations related to current_user"""
    return await conversation_repository.get_sessions(current_user.user_id, db)


@router.get("/messages", response_model=List[MessageResponse|None])
async def get_messages(
    session_id: uuid.UUID,
    current_user: User = Depends(get_current_user),
    db: AsyncSession = Depends(get_db),
) -> List[Message]:
    """get the list of messages to a given valid session_id"""
    await conversation_repository.verify_session(
        current_user.user_id, session_id, db)
    
    return await conversation_repository.get_messages(session_id, db)
