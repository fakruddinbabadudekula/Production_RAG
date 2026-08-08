"""Module for authentication.
contains login,refresh and logout routers"""

from fastapi import APIRouter, Depends, Response, status, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.schemas.auth import AccessTokenResponse
from app.core.config import settings
from app.schemas.auth import LoginData
from app.services.auth_service import auth_service
from app.core.db import get_db
from app.api.cookie import delete_refresh_cookie,set_refresh_cookie
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
    payload: LoginData,
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
    token_data = await auth_service.login(payload, db)
    set_refresh_cookie(response, token_data.refresh_token)
    return AccessTokenResponse(
        token=token_data.access_token, expire_at=token_data.access_expire_time
    )


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

    token_data = await auth_service.refresh(token, db)
    set_refresh_cookie(response, token_data.refresh_token)
    return AccessTokenResponse(
        token=token_data.access_token, expire_at=token_data.access_expire_time
    )


@router.post(
    "/logout",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Log out the current user",
)
async def logout(
    response: Response,
    db: AsyncSession = Depends(get_db),
    token=Cookie(alias="refresh_token"),
):
    
    await auth_service.logout(token, db)
    delete_refresh_cookie(response)

    return 
