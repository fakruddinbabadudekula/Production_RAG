from fastapi import Response, Cookie
from app.core.config import settings


def set_refresh_cookie(response: Response, token: str):
    response.set_cookie(
        key="refresh_token",
        value=token,
        path="/api/v1/auth",  # why upto /auth instead of /refresh because of we need to access cookie for /logout
        secure=False,  # For postman testing only, make True in production.
        httponly=True,
        samesite="strict",
        max_age=60
        * 60
        * 24
        * settings.REFRESH_TOKEN_EXPIRE_DAYS,  # why we multiply, because max_age accepts in seconds.
    )


def delete_refresh_cookie(response: Response):
    response.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth",
        httponly=True,
        secure=False,
        samesite="strict",
    )
