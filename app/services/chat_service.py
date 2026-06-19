from functools import lru_cache

from app.rag.workflow.graph import Graph
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from typing import List
from app.schemas.message import MessageSchema
from app.models.message import Message
from app.schemas.enums import MessageRole
from app.services.llm_client import llm_client
from app.services.vector_store_service import get_vector_path,get_vector_service
from app.repositories.conversation_repository import conversation_repository



class ChatService:
    def __init__(self):
        self.graph = Graph(llm_client=llm_client)

    def convert_msg_to_langchain_msg(
        self, messages: List[Message]
    ) -> List[BaseMessage]:
        langchain_msgs: List[BaseMessage] = []
        role_mapper = {
            MessageRole.SYSTEM: SystemMessage,
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
        }

        for message in messages:
            mapper_class = role_mapper.get(message.role)
            if not mapper_class:
                raise ValueError(f"Unsupported message role: {message.role}")
            langchain_msgs.append(mapper_class(content=message.content))
        return langchain_msgs


    async def chat(self, user_id, session_id, query, db):
        messages = await conversation_repository.get_messages(session_id, db)
        langchain_msgs = self.convert_msg_to_langchain_msg(messages)
        langchain_msgs.append(HumanMessage(content=query))
        retriever = await get_vector_service(
            get_vector_path(str(user_id), str(session_id))
        ).get_retriever()
        response = await self.graph.ainvoke(langchain_msgs, retriever)
        user_msg = MessageSchema(
            session_id=session_id, role="user", content=query, top_k_docs=None
        )

        assistant_msg = MessageSchema(
            session_id=session_id,
            role="assistant",
            content=response["response"].content,
            top_k_docs=response["top_k_docs"],
        )
        new_messages=await conversation_repository.add_messages(
            [user_msg,assistant_msg],db
        )
        return response


@lru_cache()
def get_service():
    return ChatService()


chat_service = get_service()
