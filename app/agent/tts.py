import logging
from io import BytesIO
from spitch import Spitch
from pydub import AudioSegment
from app.config.settings import dot_env_pth

logger = logging.getLogger(__name__)

client = Spitch(api_key=dot_env_pth.SPITCH_API_KEY)

LANG_VOICE_MAP = {
    "en":  "john",
    "yo":  "sade",
    "ha":  "hasan",
    "ig":  "obinna",
    "pcm": "ufoma",
}

def Text2Speech(txt: str, lang: str = "en") -> AudioSegment:
    voice = LANG_VOICE_MAP[lang]
    logger.info(f"TTS | lang={lang}, voice={voice}, chars={len(txt)}")
    response = client.speech.generate(
        language=lang,
        text=txt,
        voice=voice,
        # format="wav",
    )
    content = response.read()
    logger.info(f"TTS | bytes received: {len(content)}")  
    if not content:
        raise ValueError("TTS returned empty response — check Spitch credit balance.")
    audio = AudioSegment.from_file(BytesIO(content), format="wav")
    logger.info(f"TTS | done, duration={audio.duration_seconds:.1f}s")
    return audio

# print(Text2Speech("ẹ̀rọ àti bí a ṣe le lo kóòdù bíìtà láti dá àwọn ìṣòro sílẹ̀.", "yo"))