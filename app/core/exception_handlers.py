from fastapi import Request,FastAPI
from fastapi.responses import JSONResponse
from app.core.exceptions import AppException
import logging

logger = logging.getLogger(__name__)


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
                "code": "INTERNAL_SERVER_ERROR",
                "message": "An unexpected error occurred.",
            }
        },
    )


async def operational_exception_handler(request: Request, exc: AppException):
    message = exc.detail
    extra = exc.extra
    logger.warning(message, exc_info=exc, extra=extra)

    return JSONResponse(
        status_code=exc.http_status,
        content={
            "error": {
                "code": exc.error_code,
                "message": message,
                "extra": extra,
            }
        },
    )

def register_exception_handlers(app: FastAPI):

    app.add_exception_handler(
        AppException,
        operational_exception_handler,
    )
    app.add_exception_handler(
        Exception,
        unhandled_exception_handler,
    )