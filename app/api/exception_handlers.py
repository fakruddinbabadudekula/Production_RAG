from fastapi import Request,FastAPI
from fastapi.responses import JSONResponse
from app.core.exceptions import AppException
import logging
from fastapi.exceptions import RequestValidationError
from fastapi import status
from app.core.exceptions import *
EXCEPTION_STATUS_MAP = {
    ResourceNotFoundException: status.HTTP_404_NOT_FOUND,
    ValidationException: status.HTTP_400_BAD_REQUEST,
    UnSupportedResource: status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
    DuplicateResourceException: status.HTTP_409_CONFLICT,
    InvalidCredentialsException: status.HTTP_401_UNAUTHORIZED,
    LLMServieException: status.HTTP_503_SERVICE_UNAVAILABLE,
    InvalidFilePaths: status.HTTP_500_INTERNAL_SERVER_ERROR,
}
logger = logging.getLogger(__name__)

def serialize_validation_errors(errors):
    """serialize the validation errors into list"""
    result = []

    for err in errors:
        err = err.copy()

        if "ctx" in err:
            err["ctx"] = {
                k: str(v)
                for k, v in err["ctx"].items()
            }

        result.append(err)

    return result
async def unhandled_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.error(
        "Unhandled exception",
        exc_info=exc,
    )

    return JSONResponse(
        status_code=500,
        content={
            "error": {
                "error_type":"Internal_server_error",
                "message": "An unexpected error occurred.",
            }
        },
    )


async def validation_exception_handler(
    request,
    exc: RequestValidationError,
):
    return JSONResponse(
        status_code=422,
        content={
            "error_type": "validation_error",
            "message": "Request validation failed",
            "details": serialize_validation_errors(exc.errors()),
        },
    )

async def operational_exception_handler(request: Request, exc: AppException):
    message = exc.message
    extra = exc.details
    status_code= EXCEPTION_STATUS_MAP.get(
    type(exc),
    status.HTTP_500_INTERNAL_SERVER_ERROR,
)
    return JSONResponse(
        status_code=status_code,
        content={
            "error": {
                'error_type':exc.error_type.value,
                "message": message,
                "extra": extra,
            }
        },
    )

def register_exception_handlers(app: FastAPI):
    app.add_exception_handler(
        RequestValidationError,validation_exception_handler,
    )
    app.add_exception_handler(
        AppException,
        operational_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )