from fastapi import APIRouter
from app.schemas.auth import SignupRequest, LoginRequest, TokenResponse
from app.services import auth_service

router = APIRouter(prefix="/auth", tags=["Auth"])


@router.post("/signup", response_model=TokenResponse, status_code=201)
def signup(data: SignupRequest):
    """Register a new student. Returns a JWT access token."""
    return auth_service.signup(data)


@router.post("/login", response_model=TokenResponse)
def login(data: LoginRequest):
    """Login with email + password. Returns a JWT access token."""
    return auth_service.login(data)