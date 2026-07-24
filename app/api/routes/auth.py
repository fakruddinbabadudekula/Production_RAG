"""Module for authentication.
contains login,refresh and logout routers"""

from fastapi import APIRouter, Depends, Response, status, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.schemas.auth import LoginRequest, AccessTokenResponse

from app.services.auth_service import auth_service
from app.core.db import get_db
from app.api.dependencies import get_current_user

router = APIRouter()


@router.post(
    "/login",
    response_model=AccessTokenResponse,
    summary="Log in with email and password",
    responses={
        401: {"description": "Invalid email or password"},
    },
)
async def login(
    payload: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AccessTokenResponse:
    """
    Authenticate a user and issue tokens.

    - Returns an **access token** in the response body (short-lived, used as
      `Authorization: Bearer <token>` on protected routes).
    - Sets a **refresh token** as an `HttpOnly` cookie, scoped to
      `/api/v1/auth/refresh`, valid for 7 days.
    """
    access_token, refresh_token = await auth_service.login(payload, db)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        path="/api/v1/auth/refresh",
        secure=True,
        httponly=True,
        samesite="strict",
        max_age=60 * 60 * 24 * 7,
    )
    return AccessTokenResponse(token=access_token)


# I think we need to upgrade this roter for better authentication.current_user is not used and creates new values but didn't mean that they disable to work with previous values.
@router.post(
    "/refresh",
    response_model=AccessTokenResponse,
    summary="Exchange a refresh token for a new access token",
    responses={
        401: {"description": "Missing, invalid, or expired refresh token"},
    },
)
async def refresh(
    response: Response,
    db: AsyncSession = Depends(get_db),
    token=Cookie(alias="refresh_token"),
) -> AccessTokenResponse:
    """
    Reads the `refresh_token` cookie (not a request body) and issues a new
    access + refresh token pair, rotating the cookie.

    > **Note:** the previous refresh token is not yet invalidated server-side
    > — see the "Known Limitations" section of the README.
    """
    new_access_token, new_refresh_token = await auth_service.refresh(token, db)
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        path="/api/v1/auth/refresh",
        secure=True,
        httponly=True,
        samesite="strict",
        max_age=60 * 60 * 24 * 7,
    )
    return AccessTokenResponse(token=new_access_token)


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out the current user",
)
async def logout(response: Response, current_user=Depends(get_current_user)):
    """
    Clears the `refresh_token` cookie. Does not currently invalidate the
    access token already in the client's possession (it simply expires
    naturally per `ACCESS_TOKEN_EXPIRE_MINUTES`).
    """
    response.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth/refresh",
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return None
