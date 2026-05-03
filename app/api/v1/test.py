from fastapi import APIRouter, Depends
from app.core.security import get_current_student
from app.schemas.test_schemas import (
    TestStartRequest, TestStartResponse, QuestionOut,
    TestSubmitRequest, TestResultResponse, QuestionResult,
)
from app.services import test_service

router = APIRouter(prefix="/test", tags=["Test"])


@router.post("/start", response_model=TestStartResponse, status_code=201)
def start_test(data: TestStartRequest, student=Depends(get_current_student)):
    """
    Generate a fresh test for the given chapter.
    Returns attempt_id + list of questions (no correct answers exposed).
    """
    result = test_service.start_test(
        chapter_id=data.chapter_id,
        student_id=student["id"],
        difficulty=data.difficulty,
        topic=data.topic,
        num_questions=data.num_questions,
    )
    questions_out = [
        QuestionOut(
            id=q.id,
            question_text=q.question_text,
            options=q.options,
            difficulty=q.difficulty.value,
            keywords=q.keywords,
        )
        for q in result["questions"]
    ]
    return TestStartResponse(attempt_id=result["attempt_id"], questions=questions_out)


@router.post("/submit", response_model=TestResultResponse)
def submit_test(data: TestSubmitRequest, student=Depends(get_current_student)):
    """
    Submit answers for a test attempt.
    Returns score, per-question results, weak/strong topics, and next-action advice.
    """
    answers = [a.model_dump() for a in data.answers]
    result = test_service.submit_test(
        attempt_id=data.attempt_id,
        student_id=student["id"],
        answers=answers,
    )
    return TestResultResponse(
        attempt_id=result["attempt_id"],
        score=result["score"],
        total=result["total"],
        score_percent=result["score_percent"],
        strong_topics=result["strong_topics"],
        weak_topics=result["weak_topics"],
        results=[QuestionResult(**r) for r in result["results"]],
        next_action=result["next_action"],
    )