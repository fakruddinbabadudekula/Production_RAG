"""Module for custom fastapi exception handler and register them with the app.
contains a function called registration_exception_handler to regiser the handlers to the given fastapi instance
"""

from urllib import response

from fastapi import Request, FastAPI
from fastapi.responses import JSONResponse
from app.api.cookie import delete_refresh_cookie
from app.core.exceptions import AppException
import logging
from fastapi.exceptions import RequestValidationError
from fastapi import status
from app.core.exceptions import *

logger = logging.getLogger(__name__)

# http status mapper object where it maps the status code for the exception.
EXCEPTION_STATUS_MAP = {
    ResourceNotFoundException: status.HTTP_404_NOT_FOUND,
    ValidationException: status.HTTP_400_BAD_REQUEST,
    UnSupportedResource: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    DuplicateResourceException: status.HTTP_409_CONFLICT,
    InvalidCredentialsException: status.HTTP_401_UNAUTHORIZED,
    LLMServieException: status.HTTP_503_SERVICE_UNAVAILABLE,
    InvalidFilePaths: status.HTTP_500_INTERNAL_SERVER_ERROR,
    RefreshTokenReUsedDetection: status.HTTP_401_UNAUTHORIZED,
    RefreshTokenValidationException: status.HTTP_401_UNAUTHORIZED,
}


def serialize_validation_errors(exc):
    """serialize the validation error returns the list of errors"""
    errors = []
    for err in exc.errors():
        errors.append(
            {
                "field": str(err["loc"][-1]),  # last element = most specific field
                "message": err["msg"],  # human readable message from pydantic
            }
        )
    return errors


# This handler register to startlet/fastapi default server_error_handler middleware where it handles unhandled expception in the above or below layers this middleware is the outermost layer..
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
) -> JSONResponse:
    """Handler for unhandled excetpion and have the status code for 500."""
    logger.error(
        "Unhandled exception",
        exc_info=exc,
    )
    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_type": "Internal_server_error",
                "message": "An unexpected error occurred.",
            }
        },
    )


async def validation_exception_handler(
    request,
    exc: RequestValidationError,
) -> JSONResponse:
    """Handles Validataion errors for request route level"""
    # it doesn't log becuase, the error is not a bug it's a validation error where user passed something is not valid data types or data
    details = serialize_validation_errors(exc)
    return JSONResponse(
        status_code=422,
        content={
            "error_type": "validation_error",
            "message": "Request validation failed",
            "details": details,
        },
    )


async def operational_exception_handler(
    request: Request, exc: AppException
) -> JSONResponse:
    """Base Exception hanlder for AppException"""
    message = exc.message
    extra = exc.details
    status_code = EXCEPTION_STATUS_MAP.get(
        type(exc),
        status.HTTP_500_INTERNAL_SERVER_ERROR,
    )
    # it doesn't log becuase, the error is not a bug it's all about user's mistakes like invalid email or data not found

    response = JSONResponse(
        status_code=status_code,
        content={
            "error": {
                "error_type": exc.error_type.value,
                "message": message,
                "extra": extra,
            }
        },
    )

    
    if isinstance(exc, RefreshTokenValidationException):
        delete_refresh_cookie(response) #when we hit token validation error like reused or invalid token, we need to delete that invalid token in client cookie.
    return response


def register_exception_handlers(app: FastAPI):
    """add the all custom exception handlers"""
    app.add_exception_handler(
        RequestValidationError,
        validation_exception_handler,
    )
    app.add_exception_handler(
        AppException,
        operational_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )
