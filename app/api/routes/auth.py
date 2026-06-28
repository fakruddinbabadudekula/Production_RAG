"""Module for authentication.
contains login,refresh and logout routers"""

from fastapi import APIRouter, Depends, Response, status, Cookie
from sqlalchemy.ext.asyncio import AsyncSession
from app.api.schemas.auth import LoginRequest, AccessTokenResponse

from app.services.auth_service import auth_service
from app.core.db import get_db
from app.api.dependencies import get_current_user

router = APIRouter()


@router.post("/login", response_model=AccessTokenResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    db: AsyncSession = Depends(get_db),
) -> AccessTokenResponse:
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
@router.post("/refresh", response_model=AccessTokenResponse)
async def refresh(
    response: Response,
    db: AsyncSession = Depends(get_db),
    token=Cookie(alias="refresh_token"),
    current_user=Depends(get_current_user),
) -> AccessTokenResponse:
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


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(respone: Response, current_user=Depends(get_current_user)):
    """Currently we only delte the cookie later we can implement discarding access and refresh token also."""
    respone.delete_cookie(
        key="refresh_token",
        path="/api/v1/auth/refresh",
        httponly=True,
        secure=True,
        samesite="strict",
    )
    return None
