from pydantic import BaseModel,ConfigDict
import uuid
from app.schemas.enums import MessageRole
from datetime import datetime

class SessionResponse(BaseModel):
    user_id:uuid.UUID
    session_id:uuid.UUID
    title:str
    created_at:datetime
    model_config = ConfigDict(from_attributes=True)
    
class MessageResponse(BaseModel):
    session_id:uuid.UUID
    message_id:uuid.UUID
    role:MessageRole
    content:str
    created_at:datetime
    model_config = ConfigDict(from_attributes=True)