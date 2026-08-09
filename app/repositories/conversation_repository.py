"""Repository for conversation_repository"""

from functools import lru_cache
from logging import getLogger
from app.core.exceptions import InvalidCredentialsException
from app.models.session import Session
from app.models.message import Message
from sqlalchemy import select
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.message import MessageSchema
from typing import List

logger = getLogger(__name__)


class ConversationRepository:
    async def verify_session(
        self, user_id: uuid.UUID, session_id: uuid.UUID, db: AsyncSession
    ) -> None:
        """Verify session id if not raise invalidcredentialsexception"""
        result = await db.execute(
            select(Session).where(
                Session.session_id == session_id,
                Session.user_id == user_id,
            )
        )

        session = result.scalar_one_or_none()
        if session is None:
            raise InvalidCredentialsException(
                "Invalid Session Id",
                details={"user_id": str(user_id), "session_id": str(session_id)},
            )
        return None

    async def create_session(
        self,
        user_id: uuid.UUID,
        db: AsyncSession,
        title: str | None = None,
    ) -> Session:
        new_session = Session(
            title=title,
            user_id=user_id,
        )
        # No need to do try block and doesn't need to validate user_id because that user_id comes from get_user dependecy.If any error occurs that may be due to schema level so that it catches in global exception handler.
        
        db.add(new_session)
        await db.flush()
        await db.refresh(new_session)
        logger.info(
            "created_new_session",
            extra={"user_id": str(user_id), "session_id": str(new_session.session_id)},
        )
        return new_session

    async def add_messages(
        self, messages: List[MessageSchema], db: AsyncSession
    ) -> List[Message]:
        new_messages = [Message(**msg.model_dump()) for msg in messages]

        db.add_all(new_messages)
        await db.flush()
        return new_messages

    async def get_sessions(self, user_id: uuid.UUID, db: AsyncSession) -> List[Session]:
        result = await db.execute(
            select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.created_at.desc())
        )

        sessions = result.scalars().all()
        return sessions

    async def get_messages(
        self, session_id: uuid.UUID, db: AsyncSession
    ) -> List[Message]:
        result = await db.execute(
            select(Message)
            .where(Message.session_id == session_id)
            .order_by(Message.created_at.asc())
        )

        messages = result.scalars().all()
        return messages


@lru_cache()
def get_repository():
    return ConversationRepository()


conversation_repository = get_repository()
