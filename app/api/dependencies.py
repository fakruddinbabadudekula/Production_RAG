"""Module for fastapi dependencies.
contains get_current_user:which returns the current user object,
"""

from app.models.user import User
from fastapi import Depends
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError,ExpiredSignatureError
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.exceptions import InvalidCredentialsException
from sqlalchemy import select
from app.core.db import get_db
from app.core.security import decode_token
import uuid

security = HTTPBearer()


async def get_current_user(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Dependecy method to get_current_user
    Args:
        credentials:HTTPAuthorizationCredentials = contains bearer token
        db:AsyncSession = Database connection instance.
    Returns:
        current_user:User = Current User instance.
    Raises:
        InvalidCredentialsException: raises if invalid credentials,Token Expired and jwterror"""
    
    # raise when JWTError or invalid raise
    credentials_exception = InvalidCredentialsException(
        "Invalid credentials or token", details={"WWW-Authenticate": "Bearer"}
    )

    try:
        payload = decode_token(credentials.credentials)
        if payload.get("type") != "access":
            raise credentials_exception
        user_id = payload.get("sub")
        if not user_id:
            raise 
    except ExpiredSignatureError as e:
        raise InvalidCredentialsException(
            "Token expired",details={"WWW-Authenticate": "Bearer"}
        )
    except JWTError:
        raise credentials_exception

    user_id = uuid.UUID(user_id)
    statement = select(User).where(User.user_id == user_id)
    result = await db.execute(statement)
    user = result.scalars().first()
    if user==None:
        raise credentials_exception
    return user
