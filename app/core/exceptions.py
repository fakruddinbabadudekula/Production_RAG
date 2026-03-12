from pathlib import Path
from typing import Optional


class BaseError(Exception):
    """Base class for application errors."""

    def __init__(
        self,
        message: str,
        operation: str,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message)
        self.operation = operation
        self.original_error = original_error


class VectorStoreError(BaseError):
    """Vector store operation failure."""

    def __init__(
        self,
        message: str,
        operation: str,
        file_path: Optional[Path] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, operation, original_error)
        self.file_path = file_path


class GraphError(BaseError):
    """Graph execution error."""

    def __init__(
        self,
        message: str,
        operation: str,
        user_id: Optional[str] = None,
        session_id: Optional[str] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, operation, original_error)
        self.user_id = user_id
        self.session_id = session_id


class DocumentError(BaseError):
    """Document processing error."""

    def __init__(
        self,
        message: str,
        operation: str,
        file_path: Optional[Path] = None,
        original_error: Optional[Exception] = None,
    ):
        super().__init__(message, operation, original_error)
        self.file_path = file_path
