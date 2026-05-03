from fastapi import APIRouter, Depends
from typing import List
from app.core.security import get_current_student
from app.schemas.content import SubjectOut, ChapterOut
from app.services import content_service

router = APIRouter(prefix="/content", tags=["Content"])


@router.get("/standards", response_model=List[str])
def get_standards(student=Depends(get_current_student)):
    """List all available grade standards."""
    return content_service.get_standards()


@router.get("/subjects/{grade}", response_model=List[SubjectOut])
def get_subjects(grade: str, student=Depends(get_current_student)):
    """List subjects for a given grade (e.g. '9')."""
    return content_service.get_subjects(grade)


@router.get("/chapters/{subject_id}", response_model=List[ChapterOut])
def get_chapters(subject_id: str, student=Depends(get_current_student)):
    """List chapters for a given subject."""
    return content_service.get_chapters(subject_id)