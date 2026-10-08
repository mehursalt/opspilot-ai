"""Authentication API for the local demo UI."""

from __future__ import annotations

from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from app.services.auth_service import auth_service


router = APIRouter()


class LoginRequest(BaseModel):
    username: str
    password: str


class ChangePasswordRequest(BaseModel):
    old_password: str
    new_password: str


@router.post("/auth/login")
async def login(request: LoginRequest):
    try:
        data = auth_service.login(request.username.strip(), request.password)
        return {
            "status": "success",
            "message": "login success",
            "data": data,
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@router.get("/auth/me")
async def me(authorization: str = Header(default="")):
    token = _extract_token(authorization)
    try:
        return {
            "status": "success",
            "message": "current user loaded",
            "data": auth_service.verify_token(token),
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@router.post("/auth/logout")
async def logout(authorization: str = Header(default="")):
    token = _extract_token(authorization)
    try:
        auth_service.logout(token)
        return {
            "status": "success",
            "message": "logout success",
            "data": None,
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


@router.post("/auth/change-password")
async def change_password(request: ChangePasswordRequest, authorization: str = Header(default="")):
    token = _extract_token(authorization)
    try:
        user = auth_service.verify_token(token)
        auth_service.change_password(user["username"], request.old_password, request.new_password)
        return {
            "status": "success",
            "message": "password changed, please login again",
            "data": None,
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.get("/auth/sessions")
async def sessions(authorization: str = Header(default="")):
    token = _extract_token(authorization)
    try:
        user = auth_service.verify_token(token)
        return {
            "status": "success",
            "message": "sessions loaded",
            "data": auth_service.list_sessions(user["username"]),
        }
    except ValueError as e:
        raise HTTPException(status_code=401, detail=str(e)) from e


def _extract_token(authorization: str) -> str:
    if not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="缺少登录凭证")
    return authorization.removeprefix("Bearer ").strip()
