from pydantic_settings import BaseSettings, SettingsConfigDict



class Settings(BaseSettings):


    SOURCE:str = "0"
    MODEL:str = "yolov8n.pt"

    model_config = SettingsConfigDict(
        env_file = ".env",
        env_file_encoding= "utf-8",
    )

settings = Settings()