from pydantic import BaseModel
import uuid
from app.schemas.enums import MessageRole
from langchain_core.documents.base import Document
from typing import List


class MessageSchema(BaseModel):
    session_id: uuid.UUID
    role: MessageRole
    content: str
    top_k_docs: List[dict] | None = None
