from spitch import Spitch
from app.config.settings import dot_env_pth

client = Spitch(api_key=dot_env_pth.SPITCH_API_KEY)

def Speech2Text(audio_file, lang):
    response = client.speech.transcribe(
        content=open(audio_file, "rb"),
        language=lang,
        model="legacy",
    )
    return response.text
