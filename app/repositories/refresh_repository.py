import uuid
from sqlalchemy import select, update
from app.schemas.auth import RefreshSchema
from sqlalchemy.ext.asyncio import AsyncSession
from app.models.refresh_token import RefreshToken
from functools import lru_cache


class RefreshTokenRepository:

    async def add_token(self, data: RefreshSchema, db: AsyncSession) -> RefreshToken:
        new_data = RefreshToken(**data.model_dump())
        db.add(new_data)
        await db.flush()
        await db.refresh(new_data)
        return new_data

    async def get_token_by_hash_token(
        self, hashed_token: str, db: AsyncSession
    ) -> RefreshToken:
        data = await db.execute(
            select(RefreshToken).where(RefreshToken.hashed_token == hashed_token)
        )
        return data.scalar_one_or_none()

    async def revoke_all_tokens(self, family_id: uuid.UUID, db: AsyncSession):
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.family_id == family_id)
            .values(used=True)
        )

    async def mark_refresh_as_used(self, refresh_token_id: uuid.UUID, db: AsyncSession):
        await db.execute(
            update(RefreshToken)
            .where(RefreshToken.token_id == refresh_token_id)
            .values(used=True)
        )


@lru_cache()
def get_repository() -> RefreshTokenRepository:
    return RefreshTokenRepository()


refresh_repository = get_repository()
