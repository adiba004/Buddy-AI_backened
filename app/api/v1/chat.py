from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from app.core.security import get_current_student
from app.schemas.chat import SessionStartRequest, SessionOut, ChatMessageRequest
from app.services import chat_service

router = APIRouter(prefix="/chat", tags=["Chat"])


@router.post("/session", response_model=SessionOut, status_code=201)
def start_or_resume_session(
    data: SessionStartRequest,
    student=Depends(get_current_student),
):
    """
    Start a new chat session for a chapter or resume an existing one.
    One session per student per chapter — always resumes.
    """
    session = chat_service.get_or_create_session(student["id"], data.chapter_id)
    return chat_service.get_session_info(session["id"], student["id"])


@router.get("/session/{session_id}", response_model=SessionOut)
def get_session(session_id: str, student=Depends(get_current_student)):
    """Get session info including message count."""
    try:
        return chat_service.get_session_info(session_id, student["id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.post("/message")
def send_message(
    data: ChatMessageRequest,
    student=Depends(get_current_student),
):
    """
    Send a message and receive a **streamed** response (Server-Sent Events).

    Response format: plain text chunks streamed as `text/event-stream`.
    Each chunk is a token. Consume until stream closes.
    """
    try:
        # Verify session belongs to this student
        chat_service.get_session(data.session_id, student["id"])
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))

    def event_stream():
        for token in chat_service.stream_chat_response(
            session_id=data.session_id,
            student_id=student["id"],
            query=data.message,
        ):
            yield token

    return StreamingResponse(event_stream(), media_type="text/event-stream")