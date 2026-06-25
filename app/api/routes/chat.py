
from faiss import extra_wrappers
from fastapi import APIRouter, Depends
from app.api.schemas.chat import ChatRequest,ChatRespose
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user,get_db
from app.models.user import User
from app.repositories.conversation_repository import conversation_repository
from app.services.chat_service import chat_service
from app.core.exceptions import InvalidCredentialsException

router=APIRouter()
@router.post("/chat",response_model=ChatRespose)
async def chat(payload:ChatRequest,db:AsyncSession=Depends(get_db),current_user:User=Depends(get_current_user)):
    _=await conversation_repository.verify_session(current_user.user_id,payload.session_id,db)
    response =await chat_service.chat(current_user.user_id,payload.session_id,payload.query,db)
    return ChatRespose(
        query=payload.query,
        response=response['response'].content,
        top_k_docs=response['top_k_docs']
    )