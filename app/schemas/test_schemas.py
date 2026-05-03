from pydantic import BaseModel
from typing import List, Optional


class TestStartRequest(BaseModel):
    chapter_id: str
    difficulty: str = "medium"   # easy | medium | hard
    topic: Optional[str] = None  # None = all topics
    num_questions: int = 10


class QuestionOut(BaseModel):
    id: int
    question_text: str
    options: List[str]
    difficulty: str
    keywords: List[str]


class TestStartResponse(BaseModel):
    attempt_id: str
    questions: List[QuestionOut]


class AnswerSubmit(BaseModel):
    question_id: int
    selected_answer: str
    time_taken: float


class TestSubmitRequest(BaseModel):
    attempt_id: str
    answers: List[AnswerSubmit]


class QuestionResult(BaseModel):
    question_id: int
    question_text: str
    your_answer: str
    correct_answer: str
    is_correct: bool
    explanation: str


class TestResultResponse(BaseModel):
    attempt_id: str
    score: int
    total: int
    score_percent: float
    strong_topics: List[str]
    weak_topics: List[str]
    results: List[QuestionResult]
    next_action: str