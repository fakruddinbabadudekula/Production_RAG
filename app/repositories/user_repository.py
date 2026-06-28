"""Repository Module for user details"""
from functools import lru_cache
import uuid
from app.core.exceptions import DuplicateResourceException
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.repositories.transaction import transaction
from app.schemas.user import RegisterUser
from app.core.security import hash_password
from logging import getLogger

logger = getLogger(__name__)


class UserRepository:

    async def get_user_by_email(self, user_email: str, db: AsyncSession)->User|None:
        """Get the user by using user_email.

        Args:
            user_email:str = user email to be fetched.
            db:AsyncSession = connection session.
        Returns:
            user:User = User model instance, include all details(including hashed password)
        """
        result = await db.execute(select(User).where(User.email == user_email))
        # why we don't raise exception here if user not found: the upper layer(bussiness layer) take the decision weather to raise or not.
        user = result.scalars().first()
        return user

    async def get_user_by_id(self, user_id: uuid.UUID, db: AsyncSession)->User|None:
        """Get the user by using user_id

        Args:
            user_id:UUID = user id to be fetched
            db:AsyncSession = connection session

        Returns:
            user:User = User model instance, include all details(including hashed password)
        """
        result = await db.execute(select(User).where(User.user_id == user_id))
        # why we don't raise exception here if user not found: the upper layer(bussiness layer) take the decision weather to raise or not.
        user = result.scalars().first()
        return user

    async def create_user(self, user: RegisterUser, db: AsyncSession)->User|None:
        """Create new user in the database.

        Args:
            user:RegisterUser = user details from name to password
            db:AsyncSession = connection session of database

        Returns:
            new_user:User = Return User instance with all details which are stored in database(include hashed password)

        Raises:
            DuplicateResourceException: if user already exist or Integrity Constraint error
            Excepiton: Unknown error

        """
        new_user = User(
            **user.model_dump(exclude={"password"}),
            hashed_password=hash_password(user.password),
        )
        try:

            async with transaction(db):
                db.add(new_user)
        except IntegrityError as e:
            raise DuplicateResourceException(
                "user_already_exist", details={"user_email": new_user.email}
            ) from e

        await db.refresh(new_user)
        logger.info("created_new_user", extra={"user_id": str(new_user.user_id)})
        return new_user


@lru_cache()
def get_respository():
    return UserRepository()


user_repository = get_respository()
