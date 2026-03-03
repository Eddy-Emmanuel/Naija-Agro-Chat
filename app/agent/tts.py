from io import BytesIO
from spitch import Spitch
from pydub import AudioSegment
from app.config.settings import dot_env_pth

client = Spitch(api_key=dot_env_pth.SPITCH_API_KEY)

def Text2Speech(txt, lang):
    response = client.speech.generate(
        language=lang,
        text=txt,
        voice="sade",
        format="wav"
    )
    content = response.read()
    audio = AudioSegment.from_file(BytesIO(content), format="wav")
    return audio

# audio_seg = Text2Speech("Mo fẹ́ kọ́ ẹ̀kọ́ tuntun lónìí. Ẹ jọ̀ọ́, ṣe o lè ràn mí lọ́wọ́? Mo nífẹ̀ẹ́ sí ìmọ̀ nípa ìmọ̀ ẹrọ àti bí a ṣe lè lo kóòdù Python láti dá àwọn ìṣòro sílẹ̀.", "yo")
# audio_seg.export("yoruba_audio.mp3", format="mp3")