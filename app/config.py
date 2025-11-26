from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # --- База данных ---
    POSTGRES_USER: str = Field(..., alias="POSTGRES_USER")
    POSTGRES_PASSWORD: str = Field(..., alias="POSTGRES_PASSWORD")
    POSTGRES_HOST: str = Field("postgres", alias="POSTGRES_HOST")
    POSTGRES_PORT: int = Field(5432, alias="POSTGRES_PORT")
    POSTGRES_DB: str = Field("neurovision_db", alias="POSTGRES_DB")

    # --- Redis ---
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379

    # --- Токены ---
    TINKOFF_API_TOKEN: str
    TELEGRAM_BOT_TOKEN: str
    NEWS_API_KEY: str | None = None

    # --- Пути и URL ---
    API_BASE_URL: str = "http://api:8000"
    MODEL_STORAGE_PATH: str = "/app/models_storage"

    @property
    def DATABASE_URL(self) -> str:
        return f"postgresql://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"

    @property
    def REDIS_URL(self) -> str:
        return f"redis://{self.REDIS_HOST}:{self.REDIS_PORT}/0"

    class Config:
        env_file = ".env"
        extra = "ignore"


settings = Settings()
