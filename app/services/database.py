from typing import List
from functools import lru_cache
import uuid
from logging import getLogger

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError, OperationalError
from sqlalchemy.ext.asyncio import AsyncSession

from app.schemas.auth import RegisterUser, DeleteUser
from app.utils.security import hash_password, verify_password

from app.models.user import User
from app.models.session import Session
from app.models.message import Message
from app.models.file import FileMetadata

from app.core.exceptions import (
    DatabaseConnectionError,
    DuplicateResourceError,
    EntityNotFoundError,
    InvalidCredentialsError,
)

logger = getLogger(__name__)


class DataBaseService:

    async def create_user(
        self,
        user: RegisterUser,
        db: AsyncSession,
    ) -> User:

        new_user = User(
            **user.model_dump(exclude={"password"}),
            hashed_password=hash_password(user.password),
        )

        db.add(new_user)

        try:
            await db.commit()
            await db.refresh(new_user)

            logger.info(
                "created_new_user user_id=%s",
                new_user.user_id,
            )

            return new_user

        except IntegrityError as exc:
            await db.rollback()

            raise DuplicateResourceError(
                detail="User already exists",
                extra={
                    "email": new_user.email,
                },
            ) from exc

        except OperationalError as exc:
            await db.rollback()

            raise DatabaseConnectionError() from exc

    async def verify_session(
        self,
        session_id: uuid.UUID,
        current_user_id: uuid.UUID,
        db: AsyncSession,
    ) -> bool:

        try:
            session = await db.get(Session, session_id)

        except OperationalError as exc:
            raise DatabaseConnectionError() from exc

        return session is not None and session.user_id == current_user_id

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

        db.add(new_session)

        try:
            await db.commit()
            await db.refresh(new_session)

            logger.info(
                "created_session session_id=%s",
                new_session.session_id,
            )

            return new_session

        except IntegrityError as exc:
            await db.rollback()

            raise DuplicateResourceError(
                detail="Session creation failed",
                extra={
                    "user_id": str(user_id),
                },
            ) from exc

        except OperationalError as exc:
            await db.rollback()

            raise DatabaseConnectionError() from exc

    async def get_sessions(
        self,
        user_id: uuid.UUID,
        db: AsyncSession,
    ) -> List[Session]:

        try:
            result = await db.execute(
                select(Session)
                .where(Session.user_id == user_id)
                .order_by(Session.created_at.desc())
            )

            sessions = result.scalars().all()

            logger.info(
                "retrieved_sessions user_id=%s count=%s",
                user_id,
                len(sessions),
            )

            return sessions

        except OperationalError as exc:
            raise DatabaseConnectionError() from exc

    async def get_messages(
        self,
        session_id: uuid.UUID,
        db: AsyncSession,
    ) -> List[Message]:

        try:
            result = await db.execute(
                select(Message)
                .where(Message.session_id == session_id)
                .order_by(Message.created_at.desc())
            )

            messages = result.scalars().all()

            logger.info(
                "retrieved_messages session_id=%s count=%s",
                session_id,
                len(messages),
            )

            return messages

        except OperationalError as exc:
            raise DatabaseConnectionError() from exc

    @staticmethod
    async def __get_user(
        user_email: str,
        db: AsyncSession,
    ) -> User:

        try:
            result = await db.execute(select(User).where(User.email == user_email))

            user = result.scalars().first()

            if user is None:
                raise EntityNotFoundError(
                    detail="User not found",
                    extra={
                        "email": user_email,
                    },
                )

            return user

        except OperationalError as exc:
            raise DatabaseConnectionError() from exc

    async def get_user(
        self,
        user_email: str,
        db: AsyncSession,
    ) -> User:

        return await self.__get_user(
            user_email=user_email,
            db=db,
        )

    async def get_all_users(
        self,
        db: AsyncSession,
    ) -> List[User]:

        try:
            result = await db.execute(select(User).order_by(User.created_at))

            return result.scalars().all()

        except OperationalError as exc:
            raise DatabaseConnectionError() from exc

    async def delete_user(
        self,
        user_data: DeleteUser,
        db: AsyncSession,
    ) -> User:

        user = await self.__get_user(
            user_data.email,
            db,
        )

        if not verify_password(
            user_data.password,
            user.hashed_password,
        ):
            raise InvalidCredentialsError(
                detail="Invalid credentials",
                extra={
                    "email": user_data.email,
                },
            )

        try:
            await db.delete(user)
            await db.commit()

            logger.info(
                "deleted_user user_id=%s",
                user.user_id,
            )

            return user

        except OperationalError as exc:
            await db.rollback()

            raise DatabaseConnectionError() from exc

    async def add_file(
        self,
        file_id: uuid.UUID,
        name: str,
        type: str,
        session_id: uuid.UUID,
        db: AsyncSession,
    ) -> FileMetadata:

        new_file = FileMetadata(
            file_id=file_id,
            name=name,
            type=type,
            session_id=session_id,
        )

        db.add(new_file)

        try:
            await db.commit()
            await db.refresh(new_file)

            logger.info(
                "created_file_metadata file_id=%s",
                file_id,
            )

            return new_file

        except IntegrityError as exc:
            await db.rollback()

            raise DuplicateResourceError(
                detail="File metadata already exists",
                extra={
                    "file_id": str(file_id),
                },
            ) from exc

        except OperationalError as exc:
            await db.rollback()

            raise DatabaseConnectionError() from exc


@lru_cache(maxsize=1)
def get_db_service() -> DataBaseService:
    return DataBaseService()


db_service = get_db_service()
