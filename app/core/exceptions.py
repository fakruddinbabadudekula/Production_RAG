from pathlib import Path
from typing import Optional,Any
import uuid
from app.schemas.enums import ErrorType


class BaseError(Exception):
    """Base class for application errors."""

    def __init__(
        self,
        message: str,
        operation: str,
        user_id:uuid.UUID,
        session_id:Optional[uuid.UUID|None]=None 
         
    ):
        super().__init__(message)
        self.operation = operation
        self.session_id=session_id
        self.user_id=user_id
        self.message=message


class VectorStoreError(BaseError):
    """Vector store operation failure."""

    def __init__(
        self,
        message: str,
        operation: str,
        vector_dir: Optional[Path] = None,
        **kwargs
    ):
        super().__init__(message, operation, **kwargs)
        self.vector_dir = vector_dir


class GraphError(BaseError):
    """Graph execution error."""

    def __init__(
        self,
        message: str,
        operation: str,
        step:str,
        **kwargs
    ):
        super().__init__(message, operation,**kwargs )
        self.step=step


class DocumentError(BaseError):
    """Document processing error."""

    def __init__(
        self,
        message: str,
        operation: str,
        file_path: Optional[Path] = None,
        file_type:Optional[str]=None,
        file_id:Optional[str]=None,
        **kwargs
    ):
        super().__init__(message, operation, **kwargs)
        self.file_path = file_path
        self.file_type=file_type
        self.file_id=file_id

class DatabaseError(BaseError):
    """DataBaser Serviec Error class """
    def __init__(
        self,
        message: str,
        operation: str,
        service:Optional[str]=None,
        error_type:Optional[ErrorType]=None,
        **kwargs
    ):
        super().__init__(message, operation, **kwargs)
        self.service=service
        self.error_type=error_type
        
class UploadFileError(BaseError):
    """Uploading File Error"""
    def __init__(
        self,
        message: str,
        operation: str,
        file_path: Optional[Path] = None,
        file_type:Optional[str]=None,
        file_id:Optional[str]=None,
        error_type:Optional[ErrorType]=None,
        **kwargs
    ):
        super().__init__(message, operation, **kwargs)
        self.file_path = file_path
        self.file_type=file_type
        self.file_id=file_id
        self.error_type=error_type