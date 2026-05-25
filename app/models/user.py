from __future__ import annotations
from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column,relationship
from datetime import datetime, timezone
from sqlalchemy import String,DateTime,func,Text
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.session import Session



class User(Base):
    __tablename__="users"
    user_id:Mapped[uuid.UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        default=uuid.uuid4,
        unique=True,
        index=True,
        nullable=False,
        primary_key=True
    )
    name: Mapped[str]=mapped_column(String(40),nullable=False)
    email:Mapped[str]=mapped_column(String(40),nullable=False,unique=True)
    hashed_password:Mapped[str]=mapped_column(Text,nullable=False)
    created_at:Mapped[DateTime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    
    sessions: Mapped[list["Session"]] = relationship(
        back_populates="user",
        cascade="all, delete-orphan",
    )