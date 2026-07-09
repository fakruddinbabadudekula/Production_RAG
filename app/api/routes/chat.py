"""Module for conversation routers
contains chat router which is async without streaming"""

from fastapi import APIRouter, Depends
from app.api.schemas.chat import ChatRequest, ChatRespose
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.dependencies import get_current_user, get_db
from app.models.user import User
from app.repositories.conversation_repository import conversation_repository
from app.services.chat_service import chat_service

router = APIRouter()


@router.post(
    "/chat",
    response_model=ChatRespose,
    summary="Ask a question about the documents in a session",
    responses={
        404: {
            "description": "session_id does not exist or does not belong to the current user"
        },
        503: {"description": "The LLM provider failed or timed out after retries"},
    },
)
async def chat(
    payload: ChatRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> ChatRespose:
    """
    Runs the RAG pipeline: retrieves the top-k relevant chunks from the
    session's documents, then generates an answer that cites its sources
    as `[1]`, `[2]`, etc. `top_k_docs` in the response lists the chunks used.
    """
    _ = await conversation_repository.verify_session(
        current_user.user_id, payload.session_id, db
    )
    response = await chat_service.chat(
        current_user.user_id, payload.session_id, payload.query, db
    )
    return ChatRespose(
        query=payload.query,
        response=response["response"].content,
        top_k_docs=response["top_k_docs"],
    )
