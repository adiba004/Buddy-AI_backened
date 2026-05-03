from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from app.core.database import supabase

bearer_scheme = HTTPBearer()


async def get_current_student(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> dict:
    """
    Validate Supabase JWT and return the student row.
    Raises 401 if token is invalid or student profile not found.
    """
    token = credentials.credentials
    try:
        # Supabase validates the JWT and returns the user
        user_response = supabase.auth.get_user(token)
        user = user_response.user
        if not user:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid token")
    except Exception:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid or expired token")

    # Fetch student profile
    result = (
        supabase.table("students")
        .select("*")
        .eq("id", user.id)
        .single()
        .execute()
    )
    if not result.data:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Student profile not found")

    # Update last_active_at
    supabase.table("students").update(
        {"last_active_at": "now()"}
    ).eq("id", user.id).execute()

    return result.data