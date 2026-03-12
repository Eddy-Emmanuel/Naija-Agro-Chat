import os
import streamlit as st
from io import BytesIO
import tempfile
import logging

# Suppress noisy aioice/aiortc teardown errors if those libs happen to be installed
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("streamlit_webrtc").setLevel(logging.CRITICAL)

from app.agent import NaijaAgroChat

st.set_page_config(page_title="NaijaAgroChat", layout="wide")


# ── Bot singleton ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_bot():
    return NaijaAgroChat.build(use_agent=True)


bot = get_bot()


# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history": [],
        "pending_voice_query": None,   # voice query to auto-submit on next run
        "last_audio_key": None,        # tracks which audio upload we last processed
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Helpers ───────────────────────────────────────────────────────────────────

def _append_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def _render_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _audio_bytes_to_wav(audio_bytes: bytes, src_format: str = "wav") -> str:
    """
    Write audio bytes to a temp WAV file, converting if necessary.
    Returns path to a WAV file. Requires ffmpeg for non-wav formats.
    """
    if src_format in ("wav", "wave"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as f:
            f.write(audio_bytes)
            return f.name

    # Convert via pydub
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{src_format}") as f:
        f.write(audio_bytes)
        src_path = f.name

    wav_path = src_path.rsplit(".", 1)[0] + ".wav"
    try:
        AudioSegment.from_file(src_path, format=src_format).export(wav_path, format="wav")
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass

    return wav_path


def _run_query(query: str):
    """Run a text query through the bot and append results to chat history."""
    # Pass prior conversation to the bot so it can handle follow-up questions.
    history = st.session_state.chat_history.copy()
    _append_message("user", query)

    with st.spinner("Retrieving & generating answer..."):
        try:
            result = bot.ask(query, history=history)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            return

    answer = result.get("answer", "<no answer returned>")
    _append_message("assistant", answer)

    if result.get("safe") is not None:
        _append_message("assistant", f"**Safety check:** {result['safe']}")

    with st.spinner("Generating audio response..."):
        try:
            lang = result.get("lang", "en")
            audio_seg = bot.text_to_speech(answer, lang=lang)
            buf = BytesIO()
            audio_seg.export(buf, format="mp3")
            st.audio(buf.getvalue(), format="audio/mp3")
        except Exception as exc:
            st.warning(f"TTS failed (answer still shown above): {exc}")


def _transcribe_audio(audio_bytes: bytes, src_format: str = "wav") -> str | None:
    """Convert audio to WAV and transcribe via bot.ask_audio. Returns query string or None."""
    with st.spinner("Converting audio..."):
        try:
            wav_path = _audio_bytes_to_wav(audio_bytes, src_format)
        except Exception as exc:
            st.error(
                f"Audio conversion failed: {exc}. "
                "Make sure ffmpeg is installed (add `ffmpeg` to packages.txt on Streamlit Cloud)."
            )
            return None

    with st.spinner("Transcribing audio..."):
        try:
            result, _ = bot.ask_audio(
                wav_path,
                lang="auto",
                history=st.session_state.chat_history.copy(),
            )
        except Exception as exc:
            st.error(f"Transcription failed: {exc}")
            return None
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    transcribed = (result.get("query") or "").strip()
    if not transcribed:
        st.warning("No speech detected — please try again.")
        return None

    return transcribed


# ── Detect st.audio_input availability ───────────────────────────────────────
# st.audio_input was added in Streamlit 1.32.
# We check at runtime so the app still works on older deployments.

_HAS_AUDIO_INPUT = hasattr(st, "audio_input")


# ── Page layout ───────────────────────────────────────────────────────────────

st.title("NaijaAgroChat 🔆")

st.markdown(
    """
    <style>
    body, .stApp { background: #0b0e14 !important; color: #e0e6ff !important; }
    .stTextInput > div > input {
        background: #0f1220 !important;
        color: #e0e6ff !important;
        border: 1px solid #2e3a58 !important;
    }
    .stButton button {
        background: #2563eb !important;
        color: white !important;
        border-radius: 0.75rem !important;
        height: 2.4rem !important;
    }
    .stButton button:hover { background: #1d4ed8 !important; }
    .stMarkdown { color: #e0e6ff !important; }
    .stSpinner > div { background-color: rgba(255,255,255,0.92) !important; color:#000 !important; }
    .stSpinner > div * { color: #000 !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    "Use the chat below to ask a question in text or record audio; "
    "the assistant will detect your language and reply in the same language."
)

history_container = st.container()

# ── Auto-submit a pending voice query ────────────────────────────────────────
if st.session_state.pending_voice_query:
    pending = st.session_state.pending_voice_query
    st.session_state.pending_voice_query = None
    _run_query(pending)
    st.rerun()

# ── Text input + Send ─────────────────────────────────────────────────────────
input_col, btn_col = st.columns([0.82, 0.18])
prompt = input_col.text_input(
    "Message",
    key="prompt",
    placeholder="Ask anything...",
    label_visibility="collapsed",
)
with btn_col:
    send = st.button("Send", use_container_width=True)

if send and prompt:
    _run_query(prompt)

# ── Voice input ───────────────────────────────────────────────────────────────
st.markdown("---")

if _HAS_AUDIO_INPUT:
    # ── Path A: st.audio_input (Streamlit >= 1.32) ───────────────────────────
    # Native browser mic recorder — no WebRTC peer connection, no custom JS.
    # Returns an UploadedFile (WAV) the moment the user stops recording.
    audio_value = st.audio_input(
        "🎙️ Record a voice question",
        key="audio_recorder",
        help="Click the mic, speak, then click stop. Submits automatically.",
    )

    if audio_value is not None:
        # Dedup: only process a clip once even if Streamlit re-renders
        audio_id = getattr(audio_value, "file_id", None) or id(audio_value)

        if audio_id != st.session_state.last_audio_key:
            st.session_state.last_audio_key = audio_id
            audio_bytes = audio_value.read()

            # st.audio_input always returns WAV
            transcribed = _transcribe_audio(audio_bytes, src_format="wav")
            if transcribed:
                st.session_state.pending_voice_query = transcribed
                st.rerun()

else:
    # ── Path B: File uploader fallback (Streamlit < 1.32) ────────────────────
    st.markdown("**🎙️ Upload a voice recording (WAV / WebM / MP3 / OGG / M4A):**")
    uploaded = st.file_uploader(
        "Upload audio",
        type=["wav", "webm", "mp3", "ogg", "m4a"],
        key="audio_uploader",
        label_visibility="collapsed",
    )

    if uploaded is not None:
        upload_id = uploaded.file_id
        if upload_id != st.session_state.last_audio_key:
            st.session_state.last_audio_key = upload_id
            ext = uploaded.name.rsplit(".", 1)[-1].lower() if "." in uploaded.name else "wav"
            audio_bytes = uploaded.read()

            transcribed = _transcribe_audio(audio_bytes, src_format=ext)
            if transcribed:
                st.session_state.pending_voice_query = transcribed
                st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
with history_container:
    _render_history()