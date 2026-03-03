from pydantic_settings import BaseSettings, SettingsConfigDict

class EnV(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env")
    
    OPENAI_API_KEY:str
    SPITCH_API_KEY:str
    HUGGINGFACE_API_KEY:str

dot_env_pth = EnV()