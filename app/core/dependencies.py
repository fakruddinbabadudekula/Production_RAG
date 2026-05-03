from app.models.user import User
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from app.core.db import get_db
from app.utils.security import decode_token
import uuid
security=HTTPBearer()

async def get_current_user(credentials:HTTPAuthorizationCredentials=Depends(security),db:AsyncSession=Depends(get_db))->User:

    
    # raise when JWTError raise
    credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)
    
    try:
        payload=decode_token(credentials.credentials)
        if payload.get("type")!="access":
            raise credentials_exception
        user_id=payload.get("sub")
        if not user_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user_id=uuid.UUID(user_id)
    statement = select(User).where(User.user_id == user_id)
    result = await db.execute(statement)
    user = result.scalars().first()
    return user