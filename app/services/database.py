
from typing import Optional,List
from functools import lru_cache
from fastapi.exceptions import ResponseValidationError
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.auth import RegisterUser,DeleteUser
from app.utils.security import hash_password,verify_password
from app.models.user import User
from app.models.session import Session
from app.models.message import Message
from app.models.file import FileMetadata
from app.core.exceptions import DatabaseError
import uuid
from sqlalchemy.exc import IntegrityError
from app.schemas.enums import ErrorType
from logging import getLogger
logger=getLogger(__name__)
class DataBaseService():
    
    async def create_user(self,user:RegisterUser,db:AsyncSession):
        new_user = User(
            **user.model_dump(exclude={"password"}),
            hashed_password=hash_password(user.password),
        )
        db.add(new_user)
        try:
            await db.commit()
            await db.refresh(new_user)
            logger.info("created_new_user with user_id %s",new_user.user_id)
            return new_user
        except IntegrityError as e:
            await db.rollback()
            logger.error("failed_to_add_new_user_integrity_error with user_id %s ",new_user.user_id)
            raise DatabaseError(
                "user_already_exists",
                operation="Creating",
                service="create_user",
                error_type=ErrorType.INTIGIRITY_ERROR,
            ) from e
        except Exception as e:
            await db.rollback()
            logger.error("failed_to_add_new_user_unkown_error with user_id %s ",new_user.user_id)
            raise DatabaseError(
                "create_user_failed",
                operation="Creating",
                service="create_user",
                error_type=ErrorType.UNKOWN_ERROR
            ) from e
        
    async def verify_session(self,session_id:uuid.UUID,current_user_id:uuid.UUID,db:AsyncSession)->bool:
        try:
            session=await db.get(Session,session_id)
            logger.debug("successfully_get_the_session_by_session_id_to_verify")
        except Exception as e:
            raise DatabaseError(
                "getting_session_by_id",
                operation="Query",
                service="verify_session",
                error_type=ErrorType.UNKOWN_ERROR,
                session_id=session_id
            ) from e
        if not session or session.user_id!= current_user_id:
            return False
        else:
            return True
    async def create_session(self,user_id:uuid.UUID,db:AsyncSession,title:str=None)->Session:
        new_session = Session(title=title, user_id=user_id)
        db.add(new_session)
        try:
            await db.commit()
            await db.refresh(new_session)
            logger.info("new_session_created with session_id %s",new_session.session_id)
            return new_session
        except Exception as e:
            await db.rollback()
            raise DatabaseError(
                "create_session_failed",
                operation="Creating",
                user_id=user_id,
                service="create_session",
                error_type=ErrorType.UNKOWN_ERROR,
            ) from e
            
            
    async def get_sessions(self,user_id:uuid.UUID,db:AsyncSession)->List[Session]:
        try:
            
            result=await db.execute(
                select(Session)
            .where(Session.user_id == user_id)
            .order_by(Session.created_at.desc())
            )
            response=result.scalars().all()
            logger.info("successfully_get_the_sessions for user_id %s no.of sessions %s",user_id,len(response))
            return response
        except Exception as e:
            error_msg=f"failed_get_sessions error_type_{str(e)}"
            logger.error(error_msg)
            raise DatabaseError(
                "failed_get_sessions",
                operation="Quering",
                user_id=user_id,
                service="get_sessions",
                error_type=ErrorType.UNKOWN_ERROR
            ) from e
    async def get_messages(self,session_id:uuid.UUID,db:AsyncSession)->List[Message]:
        
        try:
            result=await db.execute(
            select(Message)
        .where(Message.session_id == session_id)
        .order_by(Message.created_at.desc())
        )
            response=result.scalars().all()
            logger.info("successfully_get_the_messages for session_id %s no.of messages %s",session_id,len(response))
            return response
        except Exception as e:
            error_msg=f"failed_get_messages error_type_{str(e)}"
            logger.error(error_msg)
            raise DatabaseError(
                "getting_messages_failed",
                operation="Quering",
                session_id=session_id,
                service="get_messages",
                error_type=ErrorType.UNKOWN_ERROR
            ) from e
    @staticmethod
    async def __get_user(user_email:str,db:AsyncSession):
        
        
        try:
            statement=select(User).where(User.email == user_email)
            result=await db.execute(statement=statement)
            response=result.scalars().first()
            logger.info("successfully_get_the_user")
            return response
        except Exception as e:
            logger.info("getting_user_failed")
            raise DatabaseError(
                f"getting_user_failed",
                operation="Quering",
                service="get_user",
                error_type=ErrorType.UNKOWN_ERROR
            ) from e
           
    async def get_all_users(self,db:AsyncSession):
        
        try:
            statement=select(User).order_by(User.created_at)
            result=await db.execute(statement=statement)
            response=result.scalars().all()
            logger.info("successfully_get_all_users no.of users %s",len(response))
            return response
        except Exception as e:
            logger.error("failed_to_get_all_users")
            raise DatabaseError(
                f"failed_getting_all_users",
                operation="Quering",
                service="get_all_users",
                error_type=ErrorType.UNKOWN_ERROR
            ) from e
        
    async def get_user(self,user_email:str,db:AsyncSession):
        result= await self.__get_user(user_email,db)
        return result
        
   
    async def delete_user(self,user_data:DeleteUser,db:AsyncSession):
        user_email=user_data.email
        user=await self.__get_user(user_email,db)
        
        if user and verify_password(user_data.password,user.hashed_password):
            await db.delete(user)
            await db.commit()
            logger.info("succefully_delete_the_user")
            return user
        logger.error("invalid_credentials_while_deleting_the_user")
        raise DatabaseError(
            "incorrect_credentials",
            operation="Deleting_user",
            service="delete_user",
            error_type=ErrorType.INCORRECT_CREDENTIALS
        )
    async def add_file(self,file_id:uuid.UUID,name:str,type:str,session_id:uuid.UUID,db:AsyncSession):
        new_file=FileMetadata(type=type,session_id=session_id,file_id=file_id,name=name)
        db.add(new_file)
        try:
            await db.commit()
            await db.refresh(new_file)
            logger.info("successfully_add_the_new_file_metadata for file_id %s and type %s",file_id,type)
            return new_file
        except Exception as e:
            await db.rollback()
            logger.error("failed_to_store_file_metadata file_id %s file_type %s",file_id,type)
            raise DatabaseError(
                "failed_to_add_new_file",
                operation="Creating",
                service="add_file",
                error_type=ErrorType.UNKOWN_ERROR
            )
        
    

@lru_cache(maxsize=1)
def get_db_service() -> DataBaseService:
    return DataBaseService()


# For stateless reusable across the files, intialize only once.
db_service=get_db_service()