"""
Test service: question generation, answer evaluation, Supabase persistence.
"""

import json
import logging
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from functools import wraps
from typing import Dict, List, Optional
import time

from openai import OpenAI

from app.core.config import settings
from app.core.database import supabase

logger = logging.getLogger(__name__)

_llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.Openrouter_API_KEY)


# ─────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────

def _retry(max_attempts: int = 5, delay: float = 2.0):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_attempts):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_attempts - 1:
                        raise
                    wait = delay * (2 ** attempt)
                    logger.warning(f"Attempt {attempt+1} failed, retrying in {wait}s: {e}")
                    time.sleep(wait)
        return wrapper
    return decorator


def _llm_call(prompt: str, max_tokens: int = 1000, temperature: float = 0.7) -> str:
    response = _llm.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    return response.choices[0].message.content


# ─────────────────────────────────────────────
# Data classes
# ─────────────────────────────────────────────

class QuestionType(Enum):
    MCQ = "mcq"


class DifficultyLevel(Enum):
    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


@dataclass
class Question:
    id: int
    question_text: str
    question_type: QuestionType
    options: List[str]
    correct_answer: str
    explanation: str
    difficulty: DifficultyLevel
    keywords: List[str] = field(default_factory=list)


# ─────────────────────────────────────────────
# Topic utilities
# ─────────────────────────────────────────────

def extract_topics(summary: str, max_topics: int = 5) -> List[str]:
    prompt = f"""Extract {max_topics} clear topic names from this chapter summary.
Output ONLY a comma-separated list of short topic names (2-4 words each). No explanation.

Summary:
{summary}"""
    try:
        text = _llm_call(prompt, max_tokens=256, temperature=0.2).strip()
        return [t.strip() for t in text.split(",") if t.strip()][:max_topics]
    except Exception as e:
        logger.warning(f"Topic extraction failed: {e}")
        return []


def summarize_topics(raw_topics: List[str], max_topics: int = 3) -> List[str]:
    if not raw_topics:
        return []
    prompt = f"""Group these keywords into {max_topics} broad conceptual topics (2-4 words each).
Keywords: {', '.join(raw_topics)}
Output ONLY a comma-separated list. No explanation."""
    try:
        text = _llm_call(prompt, max_tokens=60, temperature=0.2).strip()
        return [t.strip().lstrip('-').strip() for t in text.split(",") if t.strip()][:max_topics]
    except Exception as e:
        logger.warning(f"Topic summarization failed: {e}")
        return raw_topics[:max_topics]


# ─────────────────────────────────────────────
# Question generator
# ─────────────────────────────────────────────

@_retry(max_attempts=5, delay=2.0)
def generate_questions(summary: str, num_questions: int, difficulty: str) -> List[Question]:
    prompt = f"""You are a Class 9 teacher creating test questions.

Generate exactly {num_questions} MCQ questions from this chapter summary.

REQUIREMENTS:
1. Output ONLY valid JSON — no extra text, no markdown fences.
2. Every question needs: id, question_text, question_type, options, correct_answer, explanation, difficulty, keywords
3. question_type: "mcq" only
4. difficulty: "{difficulty}" for all questions
5. options: exactly 4 items; correct_answer must exactly match one option
6. keywords: 2-3 topic tags

JSON FORMAT:
{{
  "questions": [
    {{
      "id": 1,
      "question_text": "...",
      "question_type": "mcq",
      "options": ["Option A", "Option B", "Option C", "Option D"],
      "correct_answer": "Option A",
      "explanation": "...",
      "difficulty": "{difficulty}",
      "keywords": ["topic1", "topic2"]
    }}
  ]
}}

CHAPTER SUMMARY:
{summary}

Generate exactly {num_questions} questions now:"""

    raw = _llm_call(prompt, max_tokens=5000, temperature=0.7)
    return _parse_questions(raw)


def _parse_questions(raw: str) -> List[Question]:
    json_str = raw.strip()
    if "```json" in json_str:
        json_str = json_str.split("```json")[1].split("```")[0]
    elif "```" in json_str:
        json_str = json_str.split("```")[1].split("```")[0]

    data = json.loads(json_str)
    questions = []
    for q in data.get("questions", []):
        required = ["id", "question_text", "question_type", "options", "correct_answer", "explanation", "difficulty"]
        if any(f not in q for f in required):
            logger.warning(f"Skipping Q{q.get('id', '?')} — missing fields")
            continue
        dl = str(q["difficulty"]).lower().strip()
        try:
            questions.append(Question(
                id=q["id"],
                question_text=str(q["question_text"]),
                question_type=QuestionType.MCQ,
                options=[str(o) for o in q["options"]],
                correct_answer=str(q["correct_answer"]),
                explanation=str(q["explanation"]),
                difficulty=DifficultyLevel(dl),
                keywords=[str(k) for k in q.get("keywords", [])],
            ))
        except (ValueError, KeyError) as e:
            logger.warning(f"Skipping Q{q.get('id', '?')}: {e}")

    if not questions:
        raise ValueError("No valid questions parsed from LLM response")
    return questions


# ─────────────────────────────────────────────
# Supabase: weak topics
# ─────────────────────────────────────────────

def get_weak_topics_db(chapter_id: str, student_id: Optional[str] = None) -> List[str]:
    try:
        query = (
            supabase.table("weak_topics")
            .select("topic_name,times_wrong")
            .eq("chapter_id", chapter_id)
            .eq("is_resolved", False)
            .order("times_wrong", desc=True)
            .limit(3)
        )
        if student_id:
            query = query.eq("student_id", student_id)
        return [x["topic_name"] for x in query.execute().data]
    except Exception as e:
        logger.warning(f"Weak topics fetch error: {e}")
        return []


def update_weak_topics_db(chapter_id: str, mistakes: Dict[str, int], student_id: Optional[str] = None) -> None:
    for topic, count in mistakes.items():
        try:
            q = (
                supabase.table("weak_topics")
                .select("id,times_wrong")
                .eq("chapter_id", chapter_id)
                .eq("topic_name", topic)
            )
            if student_id:
                q = q.eq("student_id", student_id)
            existing = q.execute()

            if existing.data:
                row = existing.data[0]
                supabase.table("weak_topics").update({
                    "times_wrong": row["times_wrong"] + count,
                    "last_updated_at": datetime.now(timezone.utc).isoformat(),
                }).eq("id", row["id"]).execute()
            else:
                supabase.table("weak_topics").insert({
                    "chapter_id": chapter_id,
                    "student_id": student_id,
                    "topic_name": topic,
                    "times_wrong": count,
                    "times_correct": 0,
                    "is_resolved": False,
                }).execute()
        except Exception as e:
            logger.warning(f"Weak topic update error [{topic}]: {e}")


# ─────────────────────────────────────────────
# Supabase: test attempts
# ─────────────────────────────────────────────

def create_attempt(chapter_id: str, student_id: Optional[str], total_questions: int) -> str:
    prev = (
        supabase.table("test_attempts")
        .select("attempt_number")
        .eq("chapter_id", chapter_id)
        .order("attempt_number", desc=True)
        .limit(1)
        .execute()
    )
    attempt_number = (prev.data[0]["attempt_number"] + 1) if prev.data else 1

    res = supabase.table("test_attempts").insert({
        "chapter_id": chapter_id,
        "student_id": student_id,
        "attempt_number": attempt_number,
        "status": "in_progress",
        "total_questions": total_questions,
    }).execute()
    return res.data[0]["id"]


def complete_attempt(attempt_id: str, score: int, total: int) -> None:
    supabase.table("test_attempts").update({
        "status": "completed",
        "score": score,
        "score_percent": round((score / total) * 100, 2) if total else 0,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }).eq("id", attempt_id).execute()


def update_chapter_progress(student_id: Optional[str], chapter_id: str, score_pct: float) -> None:
    if not student_id:
        return
    try:
        prog = (
            supabase.table("chapter_progress")
            .select("id,best_score_percent")
            .eq("student_id", student_id)
            .eq("chapter_id", chapter_id)
            .execute()
        )
        if prog.data:
            row = prog.data[0]
            best = max(float(row["best_score_percent"] or 0), score_pct)
            supabase.table("chapter_progress").update({
                "best_score_percent": best,
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
                "status": "completed" if score_pct >= 60 else "in_progress",
            }).eq("id", row["id"]).execute()
        else:
            supabase.table("chapter_progress").insert({
                "student_id": student_id,
                "chapter_id": chapter_id,
                "best_score_percent": score_pct,
                "status": "completed" if score_pct >= 60 else "in_progress",
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
    except Exception as e:
        logger.warning(f"chapter_progress update error: {e}")


# ─────────────────────────────────────────────
# Main test flow functions
# ─────────────────────────────────────────────

def start_test(
    chapter_id: str,
    student_id: Optional[str],
    difficulty: str,
    topic: Optional[str],
    num_questions: int,
) -> dict:
    """Generate questions, create attempt row, save questions to DB."""
    chapter = (
        supabase.table("chapters")
        .select("id,title,summary,content")
        .eq("id", chapter_id)
        .single()
        .execute()
        .data
    )
    summary = chapter.get("summary") or chapter.get("content", "")[:settings.MAX_CONTEXT_LENGTH]

    gen_summary = summary
    if topic:
        gen_summary += f"\n\nFocus ONLY on this topic: {topic}"
    gen_summary += f"\nDifficulty: {difficulty}\nGenerate MCQ questions only."

    questions = generate_questions(gen_summary, num_questions, difficulty)
    attempt_id = create_attempt(chapter_id, student_id, len(questions))

    # Save all questions to DB immediately so submit_test can find correct answers
    rows = []
    for q in questions:
        rows.append({
            "attempt_id": attempt_id,
            "question_number": int(q.id),
            "question_type": q.question_type.value,
            "question_text": q.question_text,
            "options": json.dumps(q.options),
            "correct_answer": q.correct_answer,
            "explanation": q.explanation,
            "topic_tag": ", ".join(q.keywords) if q.keywords else None,
            "student_answer": None,
            "is_correct": None,
            "answered_at": None,
        })
    supabase.table("test_questions").insert(rows).execute()

    return {
        "attempt_id": attempt_id,
        "questions": questions,
    }


def submit_test(
    attempt_id: str,
    student_id: Optional[str],
    answers: List[dict],
) -> dict:
    """Evaluate answers, persist results, return scored response."""

    # Fetch attempt
    attempt = (
        supabase.table("test_attempts")
        .select("chapter_id,total_questions")
        .eq("id", attempt_id)
        .single()
        .execute()
        .data
    )

    # Fetch stored questions to get correct answers
    q_rows = (
        supabase.table("test_questions")
        .select("*")
        .eq("attempt_id", attempt_id)
        .execute()
        .data
    )
    correct_map = {int(r["question_number"]): r for r in q_rows}

    # Evaluate answers
    correct = 0
    mistakes: Dict[str, int] = {}
    strong_set = set()
    results = []
    answer_map = {}

    for a in answers:
        qid = int(a["question_id"])
        user_ans = a["selected_answer"].strip().lower()
        row = correct_map.get(qid)
        if not row:
            continue

        is_correct = user_ans == row["correct_answer"].strip().lower()

        if is_correct:
            correct += 1
            for kw in (row.get("topic_tag") or "").split(","):
                kw = kw.strip()
                if kw:
                    strong_set.add(kw)
        else:
            for kw in (row.get("topic_tag") or "").split(","):
                kw = kw.strip()
                if kw:
                    mistakes[kw] = mistakes.get(kw, 0) + 1

        answer_map[qid] = {"answer": a["selected_answer"], "is_correct": is_correct}
        results.append({
            "question_id": qid,
            "question_text": row["question_text"],
            "your_answer": a["selected_answer"],
            "correct_answer": row["correct_answer"],
            "is_correct": is_correct,
            "explanation": row["explanation"] or "",
        })

    total = attempt["total_questions"]
    score_pct = round((correct / total) * 100, 2) if total else 0

    # Persist results
    complete_attempt(attempt_id, correct, total)

    for q_id, ans_data in answer_map.items():
        supabase.table("test_questions").update({
            "student_answer": ans_data["answer"],
            "is_correct": ans_data["is_correct"],
            "answered_at": datetime.now(timezone.utc).isoformat(),
        }).eq("attempt_id", attempt_id).eq("question_number", q_id).execute()

    update_chapter_progress(student_id, attempt["chapter_id"], score_pct)

    if mistakes:
        update_weak_topics_db(attempt["chapter_id"], mistakes, student_id)

    # Summarize topics
    final_strong = summarize_topics(list(strong_set))
    final_weak = summarize_topics(list(mistakes.keys()))

    # Next action advice
    if score_pct < 50:
        next_action = "Let's review your weak topics and try easier questions next!"
    elif score_pct < 80:
        next_action = "Good effort! Review the explanations to improve further."
    else:
        next_action = "Excellent! Ready for a harder test or the next chapter?"

    return {
        "attempt_id": attempt_id,
        "score": correct,
        "total": total,
        "score_percent": score_pct,
        "strong_topics": final_strong,
        "weak_topics": final_weak,
        "results": results,
        "next_action": next_action,
    }