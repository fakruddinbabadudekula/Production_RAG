# app/core/exceptions.py
from enum import Enum


class ErrorType(str, Enum):
    NOT_FOUND = "not_found"
    VALIDATION = "validation_error"
    CONFLICT = "conflict"
    UNAUTHORIZED = "unauthorized"
    FORBIDDEN = "forbidden"
    EXTERNAL_SERVICE = "external_service_error"
    INTERNAL = "internal_error"
    INVALID = "invalid_error"


class AppException(Exception):
    error_type: ErrorType = ErrorType.INTERNAL

    def __init__(self, message: str, *, details: dict | None = None):
        self.message = message
        self.details = details or {}
        super().__init__(message)

class LLMServieException(AppException):
    error_type=ErrorType.EXTERNAL_SERVICE
    def __init__(self, message:str="llm_client failed to generate response", *, details = None):
        super().__init__(message, details=details)
class UnSupportedResource(AppException):
    error_type= ErrorType.VALIDATION

    def __init__(self, message:str="Invalid or Unsupported Resources are provided", *, details: dict | None = None):
        super().__init__(message, details=details)

class ValidationException(AppException):
    error_type=ErrorType.VALIDATION
    def __init__(self, message:str="InValid values are provided", *, details = None):
        super().__init__(message, details=details)

class DuplicateResourceException(AppException):
    error_type = ErrorType.CONFLICT

    def __init__(
        self,
        message: str = "resource already exist",
        details: dict | None = None,
    ):
        super().__init__(
            message,
            details=details,
        )


class InvalidFilePaths(AppException):
    error_type = ErrorType.INTERNAL

    def __init__(self, message, *, details: dict | None = None):
        super().__init__(message, details=details)


class ResourceNotFoundException(AppException):
    error_type = ErrorType.NOT_FOUND

    def __init__(
        self,
        message: str = "resource not found",
        *,
        details: dict | None = None,
    ):
        super().__init__(
            message,
            details=details,
        )


class InvalidCredentialsException(AppException):
    error_type = ErrorType.UNAUTHORIZED

    def __init__(
        self, message: str = "Invalid Credentials", *, details: dict | None = None
    ):
        super().__init__(message, details=details)
