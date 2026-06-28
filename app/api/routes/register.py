"""Module for register new user.
contains only register router"""

from fastapi import APIRouter, Depends
from app.core.db import get_db
from app.schemas.user import RegisterUser
from app.api.schemas.auth import BaseUser
from app.services.auth_service import auth_service
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.user import User

router = APIRouter()


@router.post("/register", response_model=BaseUser)
async def register(payload: RegisterUser, db: AsyncSession = Depends(get_db)):
    """creates or register new user in to database"""
    return await auth_service.register(payload, db)
