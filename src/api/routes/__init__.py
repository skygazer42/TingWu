from fastapi import APIRouter
from src.api.routes.transcribe import router as transcribe_router
from src.api.routes.websocket import router as websocket_router

api_router = APIRouter()
api_router.include_router(transcribe_router)
api_router.include_router(websocket_router)

__all__ = ['api_router']
