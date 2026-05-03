from fastapi import APIRouter
from app.api.v1 import auth, content, chat, test

router = APIRouter(prefix="/api/v1")

router.include_router(auth.router)
router.include_router(content.router)
router.include_router(chat.router)
router.include_router(test.router)