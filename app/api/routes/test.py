"""Module only for test routers for now it only contains get_user.
I think this router is not usefull and get_user is don't present in this router but for now make it simple
"""

from fastapi import APIRouter, Depends

from sqlalchemy.ext.asyncio import AsyncSession
from app.api.schemas.auth import BaseUser
from app.core.db import get_db
from app.api.dependencies import get_current_user
from app.models.user import User

router = APIRouter()


@router.get("/get_user", response_model=BaseUser)
async def get_user(
    user=Depends(get_current_user), db: AsyncSession = Depends(get_db)
) -> User:
    """return the current user with the possible details."""
    return user
