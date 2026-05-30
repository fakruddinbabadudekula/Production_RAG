
from app.schemas.enums import FileType
from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column, relationship
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, Integer,String,DateTime,func,Text,Enum as SqlEnum
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.models.session import Session
    
class FileMetadata(Base):
    __tablename__="files_metadata"  
    file_id:Mapped[uuid.UUID] = mapped_column(
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
    type:Mapped[FileType]=mapped_column(
        SqlEnum(FileType)
    )
    name:Mapped[str]=mapped_column(
        String(30)
    )
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    session: Mapped["Session"] = relationship(
        back_populates="files"
    )