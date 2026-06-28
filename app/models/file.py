"""Module contain file Database Model"""
from app.schemas.enums import FileType
from app.core.db import Base
from sqlalchemy.orm import Mapped,mapped_column, relationship
from datetime import datetime, timezone
from sqlalchemy import ForeignKey, Integer,String,DateTime,func,Enum as SqlEnum
import uuid
from sqlalchemy.dialects.postgresql import UUID as PG_UUID
from typing import TYPE_CHECKING
# why we type_checking here, what type_checking returns it should be false if the file runs, then when did it be true it only true when IDE's like vscode does type checking, it's similar like compilare but type checking then only true so that for type checking only the import statement is make sence and the below usecases doesn't raise an type chekcing exception. 
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
        String()
    )
    size:Mapped[int]=mapped_column(
        Integer()
    )
    created_at:Mapped[datetime]=mapped_column(DateTime(timezone=True),
        server_default=func.now(),
        default= lambda :datetime.now(timezone.utc)  
    )
    session: Mapped["Session"] = relationship(
        back_populates="files"
    )