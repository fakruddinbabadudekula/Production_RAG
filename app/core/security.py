"""Module for Token creation and password hashing"""

from datetime import datetime, timedelta, timezone
from typing import Any

from jose import jwt
from passlib.context import CryptContext
import hashlib
from app.core.config import settings

pwd_context = CryptContext(
    schemes=[settings.HASHING_ALGO],
    deprecated="auto",
)

def hash_token(raw:str)->str:
    return hashlib.sha256(raw.encode()).hexdigest()

def hash_password(plain_password: str) -> str:
    """takes plain password and return hashed password"""
    return pwd_context.hash(plain_password)


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """takes plain and hashed password and return bool"""
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(subject: str) -> tuple[str,datetime]:
    """takes data/subject and create new access token"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {
        "sub": str(subject),
        "type": "access",
        "iat": now,
        "exp": expire,
    }
    return (
        jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM),
        expire,
    )


def create_refresh_token(subject: str | Any) -> tuple[str,datetime]:
    """takes data/subject and create new refresh token"""
    now = datetime.now(timezone.utc)
    expire = now + timedelta(days=settings.REFRESH_TOKEN_EXPIRE_DAYS)
    payload = {
        "sub": str(subject),
        "type": "refresh",
        "iat": now,
        "exp": expire,
    }
    return (
        jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM),
        expire,
    )


def decode_token(token: str) -> dict:
    """takes the token and decode it,return the dict contains like type,sub.."""
    return jwt.decode(
        token,
        settings.SECRET_KEY,
        algorithms=[settings.ALGORITHM],
    )
