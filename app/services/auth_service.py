"""Service Module for authenticate"""
from functools import lru_cache
from jose import JWTError
import uuid
from app.core.exceptions import InvalidCredentialsException
from app.models.user import User
from app.schemas.user import RegisterUser
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import user_repository
from app.api.schemas.auth import LoginRequest
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)

_DUMMY_HASH = "$argon2id$v=19$m=65536,t=3,p=4$6j2nlBJCSCkFIMTYG4MQYg$hoq39+Hpcozusp8p1h5DtCxF1MjMiHncVfzkkekPipY"


class AuthService:
    def __init__(self):
        pass

    async def register(self, payload: RegisterUser, db: AsyncSession) -> User:
        """Procces the register new user and wrapper around the create_user repository method."""
        new_user = await user_repository.create_user(payload, db)
        return new_user

    async def login(self, payload: LoginRequest, db: AsyncSession)->tuple[str, str]:
        """Process the login task.
        Args:
            payload:LoginRequest = Details which is required to create instance in User.Takes email and password
            db:AsyncSession = Database connection instance
        
        Returns:
            tuple[str,str] = access_token, refresh_token
            
        Raises: 
            InvalidCredentialsException: invalid credentials.
                                    """
        user = await user_repository.get_user_by_email(payload.email, db)
        if user == None:
            # why dummy_hash, to take costant time to return the response if user doesn't send correct email,
            # if we return immediatly then attacker know this user doesn't in the database, not the password incorrect, user doesn't know what's incorrect like email or password you return only invalid credinals.
            verify_password(payload.password, _DUMMY_HASH)
            raise InvalidCredentialsException(
                "Invalid credentials email or password",
                details={"user_email": payload.email},
            )
        if not verify_password(payload.password, user.hashed_password):
            raise InvalidCredentialsException(
                "Invalid credentials email or password",
                details={"user_email": payload.email},
            )
        access_token = create_access_token(user.user_id)
        refresh_token = create_refresh_token(user.user_id)
        return access_token, refresh_token

    async def refresh(self, token,db:AsyncSession)->tuple[str, str]:
        try:
            payload = decode_token(token)
            if payload.get("type") != "refresh":
                raise InvalidCredentialsException(
                    "Invalid credentials token",
                    details={"WWW-Authenticate": "Bearer"},
                )
            user_id = payload.get("sub")
        except JWTError:
            raise InvalidCredentialsException(
                "Invalid credentials token",
                details={"WWW-Authenticate": "Bearer"},
            )
        user = await user_repository.get_user_by_id(uuid.UUID(user_id),db)
        if not user:
            raise InvalidCredentialsException(
                "Invalid credentials token",
                details={"WWW-Authenticate": "Bearer"},
            )
        new_access_token = create_access_token(user.user_id)
        new_refresh_token = create_refresh_token(user.user_id)
        return new_access_token,new_refresh_token


@lru_cache()
def get_service():
    return AuthService()


auth_service = get_service()
