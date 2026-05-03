from fastapi import APIRouter,Depends
from app.core.db import get_db
from app.schemas.auth import BaseUser,RegisterUser
from app.utils.security import hash_password
from sqlalchemy.ext.asyncio import AsyncSession
from app.services.auth import AuthServices

router=APIRouter()
auth_services=AuthServices()

@router.post('/singup',response_model=BaseUser)
async def singup(payload:RegisterUser,db:AsyncSession=Depends(get_db)):
    new_user=await auth_services.create_user(payload,db)
    if not new_user:
        return payload
    return new_user