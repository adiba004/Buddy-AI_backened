"""
Chat service: LangGraph pipeline + Supabase session/message persistence.
Streaming is handled at the API layer — this service returns a generator.
"""

import os
from typing import Generator, List, Dict, Optional
from typing_extensions import TypedDict
from datetime import datetime, timezone

from langgraph.graph import StateGraph, END
from openai import OpenAI

from app.core.config import settings
from app.core.database import supabase

os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────
# Clients
# ─────────────────────────────────────────────

_llm = OpenAI(base_url="https://openrouter.ai/api/v1", api_key=settings.Openrouter_API_KEY)
_embed: Optional[SentenceTransformer] = None


def _get_embed_model() -> SentenceTransformer:
    global _embed
    if _embed is None:
        _embed = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embed


# ─────────────────────────────────────────────
# Session helpers
# ─────────────────────────────────────────────

def get_or_create_session(student_id: str, chapter_id: str) -> dict:
    """Resume existing session or create a new one."""
    existing = (
        supabase.table("chapter_sessions")
        .select("*")
        .eq("student_id", student_id)
        .eq("chapter_id", chapter_id)
        .execute()
    )
    if existing.data:
        session = existing.data[0]
        supabase.table("chapter_sessions").update(
            {"last_active_at": datetime.now(timezone.utc).isoformat()}
        ).eq("id", session["id"]).execute()
        return session

    # Create new session
    res = supabase.table("chapter_sessions").insert({
        "student_id": student_id,
        "chapter_id": chapter_id,
        "last_active_at": datetime.now(timezone.utc).isoformat(),
    }).execute()
    return res.data[0]


def get_session(session_id: str, student_id: str) -> dict:
    res = (
        supabase.table("chapter_sessions")
        .select("*")
        .eq("id", session_id)
        .eq("student_id", student_id)
        .single()
        .execute()
    )
    if not res.data:
        raise ValueError("Session not found or access denied")
    return res.data


def get_session_info(session_id: str, student_id: str) -> dict:
    session = get_session(session_id, student_id)
    chapter = supabase.table("chapters").select("title").eq("id", session["chapter_id"]).single().execute()
    msgs = supabase.table("messages").select("id").eq("session_id", session_id).execute()
    return {
        "session_id": session["id"],
        "chapter_id": session["chapter_id"],
        "chapter_title": chapter.data["title"],
        "created_at": session["created_at"],
        "last_active_at": session["last_active_at"],
        "message_count": len(msgs.data),
    }


def get_recent_history(session_id: str, max_turns: int = 5) -> List[Dict]:
    """Fetch last N message pairs for conversation memory."""
    res = (
        supabase.table("messages")
        .select("role,content")
        .eq("session_id", session_id)
        .order("created_at", desc=True)
        .limit(max_turns * 2)
        .execute()
    )
    return list(reversed(res.data)) if res.data else []


def save_message(session_id: str, role: str, content: str, message_type: str = "chat") -> None:
    supabase.table("messages").insert({
        "session_id": session_id,
        "role": role,
        "content": content,
        "message_type": message_type,
    }).execute()

    # Update chapter_progress message count
    try:
        session = supabase.table("chapter_sessions").select("student_id,chapter_id").eq("id", session_id).single().execute()
        s = session.data
        prog = (
            supabase.table("chapter_progress")
            .select("id,messages_count")
            .eq("student_id", s["student_id"])
            .eq("chapter_id", s["chapter_id"])
            .execute()
        )
        if prog.data:
            supabase.table("chapter_progress").update({
                "messages_count": (prog.data[0]["messages_count"] or 0) + 1,
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
            }).eq("id", prog.data[0]["id"]).execute()
        else:
            supabase.table("chapter_progress").insert({
                "student_id": s["student_id"],
                "chapter_id": s["chapter_id"],
                "status": "in_progress",
                "messages_count": 1,
                "last_accessed_at": datetime.now(timezone.utc).isoformat(),
            }).execute()
    except Exception:
        pass  # Non-critical


# ─────────────────────────────────────────────
# LangGraph state & nodes
# ─────────────────────────────────────────────

class AgentState(TypedDict):
    query: str
    chapter: dict
    history: List[Dict]
    intent: str
    context_text: str
    prompt: str


def _detect_intent(query: str) -> str:
    q = query.lower().strip()
    if any(w in q for w in ["evaluate", "check my answer", "my answer is", "is this correct", "am i right"]):
        return "evaluate"
    if any(w in q for w in ["summary", "summarize", "recap", "overview", "tl;dr"]):
        return "summary"
    return "doubt"


def _intent_node(state: AgentState) -> dict:
    return {"intent": _detect_intent(state["query"])}


def _context_node(state: AgentState) -> dict:
    chapter = state["chapter"]
    intent = state["intent"]
    query = state["query"]
    max_len = settings.MAX_CONTEXT_LENGTH

    fallback = (chapter.get("summary") or chapter.get("content", ""))[:max_len]

    if intent in ["summary", "evaluate"]:
        return {"context_text": fallback}

    # RAG for doubt
    try:
        embed = _get_embed_model()
        vec = embed.encode(query).tolist()
        res = supabase.rpc("match_chapters_filter", {
            "query_embedding": vec,
            "match_count": 3,
            "input_chapter_id": chapter["id"],
        }).execute()
        chunks = "\n\n".join(x["content_chunk"] for x in res.data) if res.data else ""
        return {"context_text": chunks or fallback}
    except Exception:
        return {"context_text": fallback}


def _prompt_node(state: AgentState) -> dict:
    query = state["query"]
    intent = state["intent"]
    context = state["context_text"]
    chapter = state["chapter"]
    history = state["history"]

    history_str = ""
    if history:
        lines = ["Previous conversation:"]
        for msg in history[-6:]:
            prefix = "Student" if msg["role"] == "user" else "Buddy AI"
            lines.append(f"{prefix}: {msg['content'][:200]}")
        history_str = "\n".join(lines)

    if intent == "doubt" and not context.strip():
        prompt = f"""You are Buddy AI. The student asked an out-of-context question.

Student Question: {query}
Current Chapter: {chapter.get('title', '')}

1. Answer VERY BRIEFLY (2-3 lines).
2. Gently remind them we are studying: {chapter.get('title', '')}.
3. Keep tone warm and friendly 😊
"""
        return {"prompt": prompt}

    prompt = f"""You are Buddy AI, a warm and friendly tutor for Indian school students.

Chapter: {chapter.get('title', '')}
DETECTED INTENT: {intent}

YOUR ONLY JOB BASED ON INTENT:

If intent is "doubt":
→ Explain in simple language with one small analogy. Under 6 lines.
→ ⛔ NO quiz, MCQ, or True/False.

If intent is "summary":
→ Clean bullet-point summary. Max 6 bullets.
→ ⛔ NO quiz, MCQ, or True/False.

If intent is "evaluate":
→ Student shared an answer. Check it and give kind feedback.
→ ⛔ NO quiz, MCQ, or True/False.

Handle Hinglish: "samaj nahi/ni" = confused, "batao/bata" = explain, "eg do" = give example.

{history_str}

Chapter Content:
{context}

Student's Message: {query}

Respond based on intent "{intent}" above. ⛔ NO quiz unless intent is "test".
"""
    return {"prompt": prompt}


def _build_graph():
    wf = StateGraph(AgentState)
    wf.add_node("intent", _intent_node)
    wf.add_node("context", _context_node)
    wf.add_node("prompt", _prompt_node)
    wf.set_entry_point("intent")
    wf.add_edge("intent", "context")
    wf.add_edge("context", "prompt")
    wf.add_edge("prompt", END)
    return wf.compile()


_graph = _build_graph()


# ─────────────────────────────────────────────
# Main streaming entry point
# ─────────────────────────────────────────────

def stream_chat_response(
    session_id: str,
    student_id: str,
    query: str,
) -> Generator[str, None, None]:
    """
    Run the LangGraph pipeline to build the prompt, then stream
    the LLM response token by token. Saves messages to Supabase after streaming.
    """
    # 1. Load session + chapter
    session = get_session(session_id, student_id)
    chapter_id = session["chapter_id"]

    chapter = supabase.table("chapters").select("id,title,summary,content").eq("id", chapter_id).single().execute().data
    history = get_recent_history(session_id, max_turns=settings.MAX_HISTORY_TURNS)

    # 2. Build prompt via LangGraph
    state = AgentState(
        query=query,
        chapter=chapter,
        history=history,
        intent="",
        context_text="",
        prompt="",
    )
    result = _graph.invoke(state)
    prompt = result["prompt"]

    # 3. Save user message
    save_message(session_id, "user", query)

    # 4. Stream LLM response
    full_response = []
    stream = _llm.chat.completions.create(
        model=settings.LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        temperature=0.7,
        stream=True,
    )
    for chunk in stream:
        delta = chunk.choices[0].delta.content
        if delta:
            full_response.append(delta)
            yield delta

    # 5. Save assistant message after stream completes
    save_message(session_id, "assistant", "".join(full_response))

    # 6. Update session last_active_at
    supabase.table("chapter_sessions").update(
        {"last_active_at": datetime.now(timezone.utc).isoformat()}
    ).eq("id", session_id).execute()