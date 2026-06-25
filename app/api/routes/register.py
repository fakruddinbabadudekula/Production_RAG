from fastapi import APIRouter,Depends
from app.core.db import get_db
from app.schemas.user import RegisterUser
from app.api.schemas.auth import BaseUser
from app.services.auth_service import auth_service
from sqlalchemy.ext.asyncio import AsyncSession


router=APIRouter()


@router.post('/singup',response_model=BaseUser)
async def singup(payload:RegisterUser,db:AsyncSession=Depends(get_db)): 
    return await auth_service.register(payload,db)