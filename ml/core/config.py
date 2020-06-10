from typing import List

from pydantic import BaseSettings, AnyHttpUrl


class Settings(BaseSettings):
    BACKEND_CORS_ORIGINS: List[AnyHttpUrl] = []
    TF_PREDICT_URL: str

    class Config:
        case_sensitive = True
        env_file = ".env"


settings = Settings()
