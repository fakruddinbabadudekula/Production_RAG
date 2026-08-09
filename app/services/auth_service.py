"""Service Module for authenticate"""

from functools import lru_cache
import uuid
from app.core.exceptions import (
    InvalidCredentialsException,
    RefreshTokenReUsedDetection,
    RefreshTokenValidationException,
)
from app.models.user import User
from app.repositories.transaction import transaction
from app.repositories.refresh_repository import refresh_repository
from app.schemas.user import RegisterUser
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.user_repository import user_repository
from app.schemas.auth import LoginData, TokenData, RefreshSchema
from datetime import datetime, timezone
from app.core.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
    hash_token,
)

_DUMMY_HASH = "$argon2id$v=19$m=65536,t=3,p=4$6j2nlBJCSCkFIMTYG4MQYg$hoq39+Hpcozusp8p1h5DtCxF1MjMiHncVfzkkekPipY"


class AuthService:
    def _generate_tokens(self, user_id: uuid.UUID) -> TokenData:
        user_id = str(user_id)
        access_token, access_expire_time = create_access_token(user_id)
        refresh_token, refresh_expire_time = create_refresh_token(user_id)
        return TokenData(
            access_token=access_token,
            refresh_token=refresh_token,
            access_expire_time=access_expire_time,
            refresh_expire_time=refresh_expire_time,
        )

    async def _verify_user_and_return_it(
        self, payload: LoginData, db: AsyncSession
    ) -> uuid.UUID:
        user = await user_repository.get_user_by_email(payload.email, db)
        if user == None:
            # We use a dummy_hash to keep the response time consistent, even when the email doesn't exist.
            # If we returned immediately, an attacker could detect that the email is not in the database based on the faster response. Instead, we always return a generic "Invalid credentials" message, so users and attackers can't tell whether the email or the password is incorrect.
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
        return user.user_id

    async def _persist_refresh_token(
        self,
        user_id: uuid.UUID,
        token_data: TokenData,
        db: AsyncSession,
        family_id: uuid.UUID = None,
    ):
        await refresh_repository.add_token(
            RefreshSchema(
                user_id=user_id,
                hashed_token=hash_token(token_data.refresh_token),
                family_id=uuid.uuid4() if family_id == None else family_id,
                expires_at=token_data.refresh_expire_time,
            ),
            db,
        )

    async def _validate_refresh_token(self, token: str, db: AsyncSession):
        stored_token = await refresh_repository.get_token_by_hash_token(
            hash_token(token), db
        )
        if stored_token is None:
            raise RefreshTokenValidationException(
                "Invalid credentials token",
                details={"WWW-Authenticate": "Bearer"},
            )
        if stored_token.used:
            # Reuse of an already-rotated token = likely theft.
            await refresh_repository.revoke_all_tokens(stored_token.family_id, db)
            raise RefreshTokenReUsedDetection(
                "Refresh token reuse detected; session revoked",
                details={"user_id": str(stored_token.user_id)},
            )
        if stored_token.expires_at < datetime.now(timezone.utc):
            raise RefreshTokenValidationException(
                "Refresh token expired",
                details={"WWW-Authenticate": "Bearer"},
            )
        return stored_token

    async def register(self, payload: RegisterUser, db: AsyncSession) -> User:
        """Procces the register new user and wrapper around the create_user repository method."""
        async with transaction(db):
            new_user = await user_repository.create_user(payload, db)
        return new_user

    async def login(self, payload: LoginData, db: AsyncSession) -> TokenData:
        async with transaction(db):
            user_id = await self._verify_user_and_return_it(payload, db)
            token_data = self._generate_tokens(user_id)
            await self._persist_refresh_token(user_id, token_data, db)
        return token_data

    async def refresh(self, token: str, db: AsyncSession) -> TokenData:
        async with transaction(db):
            stored_token = await self._validate_refresh_token(token, db)
            # we are skipping token decode here,bcz we have user_id from store_token
            user = await user_repository.get_user_by_id(stored_token.user_id, db)
            if not user:
                raise RefreshTokenValidationException(
                    "Invalid credentials token",
                    details={"WWW-Authenticate": "Bearer"},
                )
            await refresh_repository.mark_refresh_as_used(stored_token.token_id, db)
            tokens_data = self._generate_tokens(user.user_id)
            await self._persist_refresh_token(
                user.user_id, tokens_data, db, family_id=stored_token.family_id
            )
        return tokens_data

    async def logout(self, token: str, db: AsyncSession):
        async with transaction(db):
            stored_token = await self._validate_refresh_token(token, db)
            await refresh_repository.revoke_all_tokens(stored_token.family_id, db)


@lru_cache()
def get_service():
    return AuthService()


auth_service = get_service()
