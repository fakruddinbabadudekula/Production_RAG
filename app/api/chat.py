from fastapi import APIRouter, Depends
from app.models.message import Message
from app.schemas.chat import ChatRequest,ChatRespose
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user,get_db
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)

from app.schemas.enums import MessageRole
from app.models.message import Message
from typing import List
from app.models.user import User
from app.services.database import db_service
from app.core.agent.graph import graph

# these are all for test case only

router=APIRouter()

def convert_msg_to_langchain_msg(messages:List[Message])->List[BaseMessage]:
    langchain_msgs:List[BaseMessage]=[]
    role_mapper={
        MessageRole.SYSTEM:SystemMessage,
        MessageRole.USER:HumanMessage,
        MessageRole.ASSISTANT:AIMessage
    }
    
    for message in messages:
        mapper_class=role_mapper.get(message.role)
        if not mapper_class:
            raise ValueError(
                f"Unsupported message role: {message.role}"
            )
        langchain_msgs.append(mapper_class(
            content=message.content
        ))
    return langchain_msgs


@router.post("/chat")
async def chat(payload:ChatRequest,db:AsyncSession=Depends(get_db),current_user:User=Depends(get_current_user)):
    if not await db_service.verify_session(payload.session_id,current_user.user_id,db):
        raise ValueError(
            "invalid session_id"
        )
    messages=await db_service.get_messages(payload.session_id,db)
    langchain_msgs=convert_msg_to_langchain_msg(messages)
    langchain_msgs.append(HumanMessage(content=payload.query))
    response=await graph.get_response(langchain_msgs,str(current_user.user_id),str(payload.session_id))
    user_msg=Message(
        session_id=payload.session_id,
        role="user",
        content=payload.query,
        top_k_docs=None
    )
    
    assistant_msg=Message(
        session_id=payload.session_id,
        role="assistant",
        content=response['response'].content,
        top_k_docs=response["top_k_docs"]
        
    )
    db.add(user_msg)
    db.add(assistant_msg)
    await db.commit()
    return ChatRespose(
        query=payload.query,
        response=response['response'].content,
        top_k_docs=response['top_k_docs']
    )