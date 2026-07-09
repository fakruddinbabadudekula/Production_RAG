"""Module for register new user.
contains only register router"""

from fastapi import APIRouter, Depends
from app.core.db import get_db
from app.schemas.user import RegisterUser
from app.api.schemas.auth import BaseUser
from app.services.auth_service import auth_service
from sqlalchemy.ext.asyncio import AsyncSession
from fastapi import status

router = APIRouter()


@router.post(
    "/register",
    response_model=BaseUser,
    status_code=status.HTTP_201_CREATED,
    summary="Register a new user",
    responses={
        409: {"description": "A user with this email already exists"},
    },
)
async def register(payload: RegisterUser, db: AsyncSession = Depends(get_db)):
    """
    Create a new user account.

    Passwords are hashed with Argon2 before storage; the raw password is
    never persisted or returned.
    """
    return await auth_service.register(payload, db)
