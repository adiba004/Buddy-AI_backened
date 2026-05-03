from pydantic import BaseModel
from typing import Optional
from datetime import datetime


class SessionStartRequest(BaseModel):
    chapter_id: str


class SessionOut(BaseModel):
    session_id: str
    chapter_id: str
    chapter_title: str
    created_at: Optional[datetime]
    last_active_at: Optional[datetime]
    message_count: int


class ChatMessageRequest(BaseModel):
    session_id: str
    message: str


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    message_type: str
    created_at: datetime