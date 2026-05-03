from fastapi import HTTPException, status
from app.core.database import supabase
from app.schemas.auth import SignupRequest, LoginRequest, TokenResponse


def signup(data: SignupRequest) -> TokenResponse:
    """
    Create Supabase Auth user, then insert student profile row.
    The student.id is set to the auth user UUID so they match.
    """
    # 1. Create auth user
    try:
        auth_response = supabase.auth.sign_up({
            "email": data.email,
            "password": data.password,
        })
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(e))

    user = auth_response.user
    if not user:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Signup failed. Email may already be registered.",
        )

    # 2. Insert student profile (id = auth user id)
    try:
        supabase.table("students").insert({
            "id": user.id,
            "full_name": data.full_name,
            "email": data.email,
            "grade": data.grade,
            "board": data.board,
            "plan": "free",
        }).execute()
    except Exception as e:
        # Rollback: delete auth user if profile insert fails
        supabase.auth.admin.delete_user(user.id)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Profile creation failed: {e}")

    # 3. Return token
    session = auth_response.session
    return TokenResponse(
        access_token=session.access_token,
        student_id=user.id,
        full_name=data.full_name,
        grade=data.grade,
        plan="free",
    )


def login(data: LoginRequest) -> TokenResponse:
    """Authenticate with Supabase and return JWT."""
    try:
        auth_response = supabase.auth.sign_in_with_password({
            "email": data.email,
            "password": data.password,
        })
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid email or password",
        )

    user = auth_response.user
    session = auth_response.session

    if not user or not session:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Login failed")

    # Fetch student profile for response
    profile = supabase.table("students").select("full_name,grade,plan").eq("id", user.id).single().execute()
    p = profile.data

    return TokenResponse(
        access_token=session.access_token,
        student_id=user.id,
        full_name=p["full_name"],
        grade=p["grade"],
        plan=p["plan"],
    )