from __future__ import annotations
from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column, relationship
from datetime import datetime, timezone
from sqlalchemy import Boolean, ForeignKey, Integer,String,DateTime,func,Text
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from typing import TYPE_CHECKING,List, Optional
if TYPE_CHECKING:
    from app.models.user import User
    from app.models.message import Message
    from app.models.file import FileMetadata


class Session(Base):
    __tablename__="sessions"
    
    session_id:Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        index=True,
        nullable=False,
        primary_key=True
    )
    title: Mapped[Optional[str]]=mapped_column(String(30),nullable=True)
    user_id:Mapped[uuid.UUID]=mapped_column(ForeignKey("users.user_id"))
    first_prompt:Mapped[bool]=mapped_column(Boolean,default=False)
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    messages: Mapped[list["Message"]] = relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="Message.created_at",
    )
    user: Mapped["User"] = relationship(
        back_populates="sessions"
    )
    files:Mapped[List['FileMetadata']]=relationship(
        back_populates="session",
        cascade="all, delete-orphan",
        order_by="FileMetadata.created_at",
    )