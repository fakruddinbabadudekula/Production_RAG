from fastapi import APIRouter, Depends, Response, HTTPException, status, Cookie
from app.models.user import User
from sqlalchemy.ext.asyncio import AsyncSession
from app.schemas.auth import LoginRequest, AccessTokenResponse
from app.services.database import DataBaseService
from jose import JWTError
from app.core.db import get_db
from app.utils.security import (
    create_access_token,
    create_refresh_token,
    decode_token,
    verify_password,
)
from app.core.dependencies import get_current_user

router = APIRouter()
auth_serivices = DataBaseService()
_DUMMY_HASH = "$argon2id$v=19$m=65536,t=3,p=4$6j2nlBJCSCkFIMTYG4MQYg$hoq39+Hpcozusp8p1h5DtCxF1MjMiHncVfzkkekPipY"


@router.post("/login",response_model=AccessTokenResponse)
async def login(
    payload: LoginRequest,
    response: Response,
    session: AsyncSession = Depends(get_db),
):
    user = await auth_serivices.get_user(payload.email, session)
    if not user:
        # why dummy_hash, to take costant time to return the response if user doesn't send correct email,
        # if we return immediatly then attacker know this user doesn't in the database, not the password incorrect, user doesn't know what's incorrect like email or password you return only invalid credinals.
        verify_password(payload.password, _DUMMY_HASH)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    if not verify_password(payload.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    access_token = create_access_token(user.user_id)
    refresh_token = create_refresh_token(user.user_id)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        path="/api/v1/auth/refresh",
        httponly=True,
        samesite="strict",
        max_age=60 * 60 * 24 * 7,
    )
    return AccessTokenResponse(token=access_token)


@router.post("/refresh",response_model=AccessTokenResponse)
async def refresh(
    response:Response,
    db: AsyncSession = Depends(get_db),
    token=Cookie(alias="refresh_token"),
):
    credentials_exception = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Could not validate credentials",
    headers={"WWW-Authenticate": "Bearer"},
)
    
    try:
        payload=decode_token(token)
        if payload.get("type")!="refresh":
            raise credentials_exception
        user_id=payload.get("sub")
        if not user_id:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    user=await db.get(User,user_id)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid credentials"
        )
    new_access_token = create_access_token(user.user_id)
    new_refresh_token = create_refresh_token(user.user_id)
    response.set_cookie(
        key="refresh_token",
        value=new_refresh_token,
        path="/api/v1/auth/refresh",
        httponly=True,
        samesite="strict",
        max_age=60 * 60 * 24 * 7,
    )
    return AccessTokenResponse(token=new_access_token)    
        
@router.post('/logout',status_code=status.HTTP_204_NO_CONTENT)
async def logout(respone:Response,current_user=Depends(get_current_user)):
    respone.delete_cookie(
        key="refresh_token", path="/api/v1/auth/refresh",
        httponly=True, secure=True, samesite="strict",
    )
    return None