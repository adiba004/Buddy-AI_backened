from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    SUPABASE_URL: str
    SUPABASE_SERVICE_ROLE_KEY: str
    SUPABASE_JWT_SECRET: str
    Openrouter_API_KEY: str

    LLM_MODEL: str = "inclusionai/ling-2.6-1t:free"
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    MAX_HISTORY_TURNS: int = 5
    MAX_CONTEXT_LENGTH: int = 3000

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()