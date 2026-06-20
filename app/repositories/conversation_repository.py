from functools import lru_cache
from logging import getLogger
from app.models.session import Session
from app.models.message import Message
from sqlalchemy import select
import uuid
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.message import MessageSchema
from app.repositories.transaction import transaction
from typing import List

logger = getLogger(__name__)


class ConversationRepository:
    def __init__(self):
        pass

    async def verify_session(
        self, user_id: uuid.UUID, session_id: uuid.UUID, db: AsyncSession
    ) -> bool:
        session = await db.get(Session, session_id)
        return session is not None and session.user_id == user_id

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
        async with transaction(db):
            db.add(new_session)
        await db.refresh(new_session)
        logger.info(
            "created_new_session",
            extra={"user_id": str(user_id), "session_id": str(new_session.session_id)},
        )
        return new_session

    async def add_messages(
        self, messages: List[MessageSchema], db: AsyncSession
    ) -> List[Message]:
        new_messages = []
        for msg in messages:
            new_messages.append(Message(**msg.model_dump()))
        async with transaction(db):
            db.add_all(new_messages)
        # for msg in new_messages:
        #     db.refresh(msg)
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
            .order_by(Message.created_at.desc())
        )

        messages = result.scalars().all()
        return messages


@lru_cache()
def get_repository():
    return ConversationRepository()


conversation_repository = get_repository()
