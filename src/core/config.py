from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):


    SOURCE:str | int = 0
    MODEL:str = "yolov8n.pt"
    WIDTH:int = 1280
    HEIGHT:int = 640

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding= "utf-8",
    )

settings = Settings()