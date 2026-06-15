from fastapi import APIRouter,Depends
from app.services.database import DataBaseService
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.auth import BaseUser
from typing import List
from app.core.db import get_db
from app.core.dependencies import get_current_user

router=APIRouter()
auth_services=DataBaseService()

@router.get('/get_all',response_model=List[BaseUser])
async def get_all_users(db:AsyncSession=Depends(get_db)):
    return await auth_services.get_all_users(db)

@router.get('/get_user',response_model=BaseUser)
async def get_user(user=Depends(get_current_user),db:AsyncSession=Depends(get_db)):
    return user