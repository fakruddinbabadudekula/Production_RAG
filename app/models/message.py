"""Module for message database model"""
from typing import List
from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column, relationship
from datetime import datetime, timezone
from sqlalchemy import ForeignKey,DateTime,func,Text,Enum as SqlEnum,JSON
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from app.schemas.enums import MessageRole
from datetime import datetime
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
    content:Mapped[str]=mapped_column(Text)
    role:Mapped[MessageRole]=mapped_column(
        SqlEnum(MessageRole),nullable=False
    )
    
    # Here we are directly storing top_k dict where it have lenght docs and metadata of them, instead of this in future we store only doc's id's and their metadata(score), if we want doc's then take that id's then serarch the vector db.
    top_k_docs:Mapped[List|None]=mapped_column(
        JSON,nullable=True
        
    )
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    session: Mapped["Session"] = relationship(
        back_populates="messages"
    )