from pathlib import Path
from typing import Optional, Any
import uuid
from app.schemas.enums import ErrorType


class AppException(Exception):
    """
    Base exception for all application-specific and operational errors.
    """

    error_code: str = "APPLICATION_ERROR"
    http_status: int = 500
    detail: str = "An unexpected error occurred."

    def __init__(
        self,
        detail: str | None = None,
        *,  # why we use star here: it specifies from here you need to pass arguments as key_value pairs
        error_code: str | None = None,
        http_status: int | None = None,
        extra: dict[str, Any] | None = None,
    ):
        self.detail = detail or self.detail
        self.error_code = error_code or self.error_code
        self.http_status = http_status or self.http_status
        self.extra = extra or {}

        super().__init__(self.detail)


class DatabaseError(AppException):
    error_code = "DATABASE_ERROR"
    http_status = 500
    detail = "Database operation failed"


class EntityNotFoundError(AppException):
    error_code = "ENTITY_NOT_FOUND"
    http_status = 404
    detail = "Resource not found"


class DuplicateResourceError(DatabaseError):
    error_code = "DUPLICATE_RESOURCE"
    http_status = 409
    detail = "Resource already exists"


class DatabaseConnectionError(DatabaseError):
    error_code = "DATABASE_CONNECTION_ERROR"
    http_status = 503
    detail = "Database unavailable"


class InvalidCredentialsError(AppException):
    error_code = "INVALID_CREDENTIALS"
    http_status = 401
    detail = "Invalid credentials"


class InvalidFilePath(AppException):
    error_code = "INVALID_FILE_PATH"
    http_status = 400
    detail = "Invalid File Path"


class InvalidFileType(AppException):
    error_code = "INVALID_FILE_TYPE"
    http_status = 400
    detail = "Invalid File Type"


class ProcessTimeOutError(AppException):
    error_code = "ProccessTimeOutError"
    http_status = 504
    detail = "Proccess TimeOut after retries"


class DocumentProcessingError(AppException):
    error_code = ("Document_Error",)
    http_status = 500
    detail = "Document processing error"
    
class GraphError(AppException):
    """Specific for Rag Pipeline"""
    pass
