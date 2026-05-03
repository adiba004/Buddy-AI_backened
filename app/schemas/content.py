from pydantic import BaseModel
from typing import List


class SubjectOut(BaseModel):
    id: str
    name: str


class ChapterOut(BaseModel):
    id: str
    title: str