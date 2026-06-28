"""Module contains ContextVar variables
contains request_id and reqeust_route"""

from contextvars import ContextVar

# stores the request_id
request_id_ctx: ContextVar[str] = ContextVar("request_id", default=None)
# stores the request_route like api/v1/.....
request_route_ctx: ContextVar[str] = ContextVar("request_route", default=None)
