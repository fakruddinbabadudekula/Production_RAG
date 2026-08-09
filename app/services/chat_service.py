"""Service module to conversation to and from rag."""

from functools import lru_cache
import uuid

from app.core.config import settings
from app.core.exceptions import LLMServieException
from app.rag.workflow.graph import Graph
from langchain_core.messages import (
    HumanMessage,
    AIMessage,
    SystemMessage,
    BaseMessage,
)
from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from app.repositories.transaction import transaction
from app.schemas.message import MessageSchema
from app.models.message import Message
from app.schemas.enums import MessageRole
from app.services.llm_client import llm_client
from app.services.vector_store_service import get_vector_path, get_vector_service
from app.repositories.conversation_repository import conversation_repository
from app.services.llm_client import RETRYABLE_LLM_EXCEPTIONS


class ChatService:
    def __init__(self):
        self.graph = Graph(llm_client=llm_client)

    def convert_msg_to_langchain_msg(
        self, messages: List[Message]
    ) -> List[BaseMessage]:
        """Convert application messages into LangChain message objects.

        Args:
            messages:
                Conversation messages stored in the database.

        Returns:
            List[BaseMessage]:
                LangChain message objects preserving the original order.

        Raises:
            ValueError:
                If an unsupported message role is encountered.
        """
        langchain_msgs: List[BaseMessage] = []
        role_mapper = {
            MessageRole.SYSTEM: SystemMessage,
            MessageRole.USER: HumanMessage,
            MessageRole.ASSISTANT: AIMessage,
        }

        for message in messages:
            mapper_class = role_mapper.get(message.role)
            if not mapper_class:
                # This should be our fault, not the user's. The user either didn't provide the roles or provided them correctly, but we assigned the wrong role while storing the messages or mapping the roles.
                raise ValueError(f"Unsupported message role: {message.role}")
            langchain_msgs.append(mapper_class(content=message.content))
        return langchain_msgs

    async def chat(
        self, user_id: uuid.UUID, session_id: uuid.UUID, query: str, db: AsyncSession
    ) -> dict:
        """Generate an AI response for a user query.

        Retrieves the conversation history, executes the RAG workflow,
        stores the new conversation messages, and returns the generated
        response.

        Args:
            user_id:
                User identifier.

            session_id:
                Conversation session identifier.

            query:
                User's input message.

            db:
                Database session.

        Returns:
            dict:
                Dictionary containing the generated response and retrieved
                source documents.

        Raises:
            LLMServiceException:
                If the language model fails after exhausting all retries.
        """
        await conversation_repository.verify_session(user_id,session_id,db)
        messages = await conversation_repository.get_messages(session_id, db)
        
        langchain_msgs = self.convert_msg_to_langchain_msg(messages)
        langchain_msgs.append(HumanMessage(content=query))
        try:
            retriever = get_vector_service(
                get_vector_path(str(user_id), str(session_id))
            ).get_retriever()
            response = await self.graph.ainvoke(langchain_msgs, retriever)
        except RETRYABLE_LLM_EXCEPTIONS as e:
            raise LLMServieException(
                "llm_service failed to generate response after retries",
                details={
                    "retries_count": settings.MAX_LLM_CALL_RETRIES,
                    "user_id": str(user_id),
                    "session_id": str(session_id),
                },
            ) from e

        # For now, we only store successful messages. Later, we can improve this by adding a status field to the message table.
        user_msg = MessageSchema(
            session_id=session_id, role="user", content=query, top_k_docs=None
        )

        assistant_msg = MessageSchema(
            session_id=session_id,
            role="assistant",
            content=response["response"].content,
            # For now, we store top_k_docs directly. Later, we can either move them to a separate table or store only the document IDs.
            top_k_docs=response["top_k_docs"],
        )
        async with transaction(db):
            _ = await conversation_repository.add_messages([user_msg, assistant_msg], db)
        return response


@lru_cache()
def get_service():
    return ChatService()


chat_service = get_service()
