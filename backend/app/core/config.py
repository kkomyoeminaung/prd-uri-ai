from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    GEMINI_API_KEY: str = ""
    GEMINI_MODEL: str = "gemini-1.5-flash"
    APP_NAME: str = "PRD-URI AI v4"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    LOG_LEVEL: str = "INFO"
    VECTOR_STORE_PATH: str = "./memory/vector_stores"
    DATABASE_URL: str = "sqlite:///./prd_uri_v4.db"
    ALPHA_RELATIONAL: float = 1.274
    DEFAULT_SCALE_L: float = 1e-10

    class Config:
        env_file = ".env"

settings = Settings()
