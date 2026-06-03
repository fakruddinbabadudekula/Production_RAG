from logging import getLogger
import uuid
from fastapi import FastAPI
from starlette.requests import Request
from starlette.responses import Response
from app.core.contextvar import request_id_ctx, request_route_ctx
import time

logger = getLogger(__name__)


async def logger_middleware(request: Request, call_next) -> Response:
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
    try:
        response: Response = await call_next(request)
    finally:
        duration_ms = (time.perf_counter() - start_time) * 1000
        logger.info(
            "requst_completed",
            extra={
                "request_id": request_id,
                "request_route": request.url.path,
                "method": request.method,
                "time_taken": duration_ms,
            },
        )
    response.headers['X-request-id']=request_id
    return response
