"""Module to register middlewares.
contains logger middleware"""

from logging import getLogger
import uuid
from starlette.requests import Request
from starlette.responses import Response
from app.core.contextvar import request_id_ctx, request_route_ctx
import time

logger = getLogger(__name__)


async def logger_middleware(request: Request, call_next) -> Response:
    """A method that adds the reqest_id and log the start and end logging"""
    # if already exist get that if not generate new one
    request_id = request.headers.get("x-request-id") or str(uuid.uuid4())
    request_id_ctx.set(request_id)
    request_route_ctx.set(request.url.path)
    logger.info(
        "request_started",
        extra={
            "request_id": request_id,
            "request_route": request.url.path,
            "method": request.method,
        },
    )
    start_time = time.perf_counter()
    response: Response | None = None
    try:
        response = await call_next(request)
    finally:
        duration = (time.perf_counter() - start_time) 
        logger.info(
            "request_completed",
            extra={
                "request_id": request_id,
                "status_code":response.status_code if response else 500,
                "request_route": request.url.path,
                "method": request.method,
                "duration": duration,
            },
        )
    response.headers['X-request-id']=request_id
    return response
