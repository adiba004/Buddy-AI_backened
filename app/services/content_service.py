from typing import List
from app.core.database import supabase


def get_standards() -> List[str]:
    res = supabase.table("subjects").select("grade").execute()
    return sorted(set(str(x["grade"]) for x in res.data))


def get_subjects(grade: str) -> List[dict]:
    return supabase.table("subjects").select("id,name").eq("grade", int(grade)).execute().data


def get_chapters(subject_id: str) -> List[dict]:
    return supabase.table("chapters").select("id,title").eq("subject_id", subject_id).execute().data


def get_chapter(chapter_id: str) -> dict:
    res = (
        supabase.table("chapters")
        .select("id,title,summary,content")
        .eq("id", chapter_id)
        .single()
        .execute()
    )
    return res.data