from fastapi import APIRouter, Depends
from app.models.message import Message
from app.schemas.chat import ChatRequest,ChatRespose
from sqlalchemy.ext.asyncio import AsyncSession
from app.core.dependencies import get_current_user,get_db
from langchain_core.messages import HumanMessage
from app.models.user import User
from app.services.database import db_service
from app.utils.chat import convert_msg_to_langchain_msg
from app.core.agent.graph import graph

# these are all for test case only
from app.core.agent.document_loaders.doc_loader import DocumentLoader
from app.core.agent.retrievers.vector_retriever import Retriever
from app.utils.graph import get_vector_path
from pathlib import Path
router=APIRouter()

async def add_docs(user_id,session_id):
    path = "storage/data/attention is all you need.pdf"
    doc_loader = DocumentLoader()
    retriever = Retriever(
        vector_dir_path=get_vector_path(user_id,session_id)
    )
    docs = await doc_loader.process_document(file_path=Path(path))
    return await retriever.aadd_documents(docs=docs)

@router.post("/chat")
async def chat(payload:ChatRequest,db:AsyncSession=Depends(get_db),current_user:User=Depends(get_current_user)):
    if not payload.session_id:
        new_session=await db_service.create_session(user_id=current_user.user_id,db=db,title=payload.query[:16])
        await add_docs(current_user.user_id,new_session.session_id)
        payload.session_id=new_session.session_id
    if not await db_service.verify_session(payload.session_id,current_user.user_id,db):
        raise ValueError(
            "invalid session_id"
        )
    messages=await db_service.get_messages(payload.session_id,db)
    langchain_msgs=convert_msg_to_langchain_msg(messages)
    langchain_msgs.append(HumanMessage(content=payload.query))
    response=await graph.get_response(langchain_msgs,current_user.user_id,payload.session_id)
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