from functools import lru_cache
import uuid

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.models.user import User
from app.repositories.transaction import transaction
from app.schemas.auth import RegisterUser
from app.core.security import hash_password


class UserRepository:
    def __init__(self):
        pass

    async def get_user_by_email(self, user_email: str, db: AsyncSession):
        result = await db.execute(select(User).where(User.email == user_email))
        user = result.scalars().first()
        return user

    async def get_user_by_id(self, user_id: uuid.UUID, db: AsyncSession):
        result = await db.execute(select(User).where(User.user_id == user_id))
        user = result.scalars().first()
        return user
    async def create_user(self,user: RegisterUser,db:AsyncSession):
        new_user = User(
            **user.model_dump(exclude={"password"}),
            hashed_password=hash_password(user.password),
        )
        async with transaction(db):
            db.add(new_user)
        await db.refresh(new_user)
        return new_user
    
@lru_cache()
def get_respository():
    return UserRepository()

user_repository=get_respository()