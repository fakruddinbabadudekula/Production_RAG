from typing import List

from pydantic import Json

from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column, relationship
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, Integer,String,DateTime,func,Text,Enum as SqlEnum,JSON
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from app.schemas.enums import MessageRole
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.session import Session
class Message(Base):
    __tablename__="messages"  
    message_id:Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        index=True,
        nullable=False,
        primary_key=True
    )
    session_id:Mapped[uuid.UUID]=mapped_column(
        ForeignKey("sessions.session_id")
    )
    content:Mapped[Text]=mapped_column(Text)
    role:Mapped[MessageRole]=mapped_column(
        SqlEnum(MessageRole),nullable=False
    )
    top_k_docs:Mapped[List|None]=mapped_column(
        JSON,nullable=True
        
    )
    created_at:Mapped[DateTime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    session: Mapped["Session"] = relationship(
        back_populates="messages"
    )