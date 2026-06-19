
from fastapi import APIRouter, Depends
from app.schemas.chat import ChatRequest,ChatRespose
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user,get_db
from app.models.user import User
from app.repositories.conversation_repository import conversation_repository
from app.services.chat_service import chat_service

router=APIRouter()
@router.post("/chat",response_model=ChatRespose)
async def chat(payload:ChatRequest,db:AsyncSession=Depends(get_db),current_user:User=Depends(get_current_user)):
    if not await conversation_repository.verify_session(current_user.user_id,payload.session_id,db):
        raise ValueError(
            "invalid session_id"
        )
    response =await chat_service.chat(current_user.user_id,payload.session_id,payload.query,db)
    return ChatRespose(
        query=payload.query,
        response=response['response'].content,
        top_k_docs=response['top_k_docs']
    )