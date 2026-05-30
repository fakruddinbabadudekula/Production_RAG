
from typing import Optional,List
from functools import lru_cache
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.auth import RegisterUser,DeleteUser
from app.utils.security import hash_password,verify_password
from app.models.user import User
from app.models.session import Session
from app.models.message import Message
from app.models.file import FileMetadata
from fastapi import HTTPException
import uuid
class DataBaseService():
    
    async def create_user(self,user:RegisterUser,db:AsyncSession):
        if not await self.__get_user(user.email,db):
            
            new_user = User(**user.model_dump(exclude="password"),hashed_password=hash_password(user.password))
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            return new_user
        
        # but you doesn't return None you should raise an error with message of already exist.
        return None
    async def verify_session(self,session_id:uuid.UUID,current_user_id:uuid.UUID,db:AsyncSession)->bool:
        session=await db.get(Session,session_id)
        if not session or session.user_id!= current_user_id:
            return False
        else:
            return True
    async def create_session(self,user_id:uuid.UUID,db:AsyncSession,title:str=None)->Session:
        new_session=Session(title=title,user_id=user_id)
        db.add(new_session)
        await db.commit()
        await db.refresh(new_session)
        return new_session
            
    async def get_sessions(self,user_id:uuid.UUID,db:AsyncSession)->List[Session]:
        result=await db.execute(
            select(Session)
        .where(Session.user_id == user_id)
        .order_by(Session.created_at.desc())
        )
        return result.scalars().all()
    async def get_messages(self,session_id:uuid.UUID,db:AsyncSession)->List[Message]:
        result=await db.execute(
            select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        )
        return result.scalars().all()
    @staticmethod
    async def __get_user(user_email:str,db:AsyncSession):
        
        statement=select(User).where(User.email == user_email)
        result=await db.execute(statement=statement)
        return result.scalars().first()
           
    async def get_all_users(self,db:AsyncSession):
        statement=select(User).order_by(User.created_at)
        result=await db.execute(statement=statement)
        return result.scalars().all()
        
    async def get_user(self,user_email:str,db:AsyncSession):
        result= await self.__get_user(user_email,db)
        return result
        
   
    async def delete_user(self,user_data:DeleteUser,db:AsyncSession):
        user_email=user_data.email
        user=await self.__get_user(user_email,db)
        if user and verify_password(user_data.password,user.hashed_password):
            await db.delete(user)
            await db.commit()
            return user
        raise HTTPException(status_code=404,detail="User not found or invalid username or password")
    async def add_file(self,file_id:uuid.UUID,name:str,type:str,session_id:uuid.UUID,db:AsyncSession):
        new_file=FileMetadata(type=type,session_id=session_id,file_id=file_id,name=name)
        db.add(new_file)
        await db.commit()
        await db.refresh(new_file)
        return new_file
        
    

@lru_cache(maxsize=1)
def get_db_service() -> DataBaseService:
    return DataBaseService()


# For stateless reusable across the files, intialize only once.
db_service=get_db_service()