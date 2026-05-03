
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.auth import RegisterUser,DeleteUser
from app.utils.security import hash_password,verify_password
from app.models.user import User
from fastapi import HTTPException
class AuthServices():
    
    async def create_user(self,user:RegisterUser,db:AsyncSession):
        if not await self.__get_user(user.email,db):
            
            new_user = User(**user.model_dump(exclude="password"),hashed_password=hash_password(user.password))
            db.add(new_user)
            await db.commit()
            await db.refresh(new_user)
            return new_user
        
        # but you doesn't return None you should raise an error with message of already exist.
        return None
   
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