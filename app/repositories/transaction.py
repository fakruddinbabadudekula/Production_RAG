"""Transaction manager """
from contextlib import asynccontextmanager
from sqlalchemy.ext.asyncio import AsyncSession


@asynccontextmanager
async def transaction(
    db: AsyncSession,
):
    try:
        yield
        await db.commit()  
    # Now here we only put Exception base class only later we will handle the more exceptions and perform exception handler
    except Exception:
        await db.rollback()
        raise