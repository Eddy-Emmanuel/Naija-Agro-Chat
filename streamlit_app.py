import base64
import os
import streamlit as st
from io import BytesIO
import tempfile
import logging

# Suppress noisy aioice/aiortc teardown errors if those libs are installed
logging.getLogger("aioice").setLevel(logging.CRITICAL)
logging.getLogger("aiortc").setLevel(logging.CRITICAL)
logging.getLogger("streamlit_webrtc").setLevel(logging.CRITICAL)

from app.agent import NaijaAgroChat

st.set_page_config(page_title="NaijaAgroChat", layout="wide")


# ── Bot singleton ─────────────────────────────────────────────────────────────

@st.cache_resource
def get_bot():
    return NaijaAgroChat.build(use_openai_generator=True)


bot = get_bot()


# ── Session state defaults ────────────────────────────────────────────────────

def _init_state():
    defaults = {
        "chat_history": [],
        "recording": False,
        "recorded_audio": None,
        "audio_processed": False,
        "audio_transcription": "",
        "recorded_audio_format": "webm",
        # A voice query that should be auto-submitted on the next run
        "pending_voice_query": None,
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


def _parse_data_url(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


def _convert_to_wav(audio_bytes: bytes, src_format: str) -> str:
    """
    Convert audio bytes (webm / wav / etc.) to a WAV temp file.
    Returns the path to the WAV file.
    pydub requires ffmpeg to be installed for non-wav formats.
    """
    from pydub import AudioSegment

    with tempfile.NamedTemporaryFile(delete=False, suffix=f".{src_format}") as tmp_in:
        tmp_in.write(audio_bytes)
        src_path = tmp_in.name

    wav_path = src_path.rsplit(".", 1)[0] + ".wav"

    if src_format == "wav":
        return src_path  # already wav, no conversion needed

    try:
        seg = AudioSegment.from_file(src_path, format=src_format)
        seg.export(wav_path, format="wav")
    finally:
        try:
            os.remove(src_path)
        except OSError:
            pass

    return wav_path


def _run_query(query: str):
    """Run a query through the bot and append results to chat history."""
    _append_message("user", query)

    with st.spinner("Retrieving & generating answer..."):
        try:
            result = bot.ask(query)
        except Exception as exc:
            st.error(f"Pipeline error: {exc}")
            st.stop()

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


# ── Pure-JS in-browser recorder ───────────────────────────────────────────────
# Uses the browser's MediaRecorder API and sends the blob back via
# Streamlit.setComponentValue — no WebRTC peer connection required.

def _audio_recorder() -> bytes | None:
    """
    Render start/stop buttons in the browser.
    Returns raw WebM bytes when a recording is ready, otherwise None.
    """
    recorded = st.components.v1.html(
        """
        <style>
            .rec-btn {
                padding: 0.35rem 0.9rem;
                border: none;
                border-radius: 0.5rem;
                cursor: pointer;
                font-size: 0.9rem;
                font-weight: 600;
                margin-right: 0.5rem;
            }
            #btn-start { background: #2563eb; color: #fff; }
            #btn-start:hover { background: #1d4ed8; }
            #btn-stop  { background: #dc2626; color: #fff; }
            #btn-stop:hover  { background: #b91c1c; }
            #btn-stop:disabled { background: #555; cursor: not-allowed; }

            .recording-dot {
                display: inline-block;
                width: 0.65rem; height: 0.65rem;
                border-radius: 50%;
                background: #ff4b4b;
                margin-right: 0.4rem;
                vertical-align: middle;
                animation: pulse 1.2s ease-in-out infinite;
            }
            @keyframes pulse {
                0%   { transform: scale(1);    opacity: 0.9; }
                50%  { transform: scale(1.35); opacity: 0.5; }
                100% { transform: scale(1);    opacity: 0.9; }
            }
        </style>

        <button id="btn-start" class="rec-btn" onclick="startRecording()">▶ Start</button>
        <button id="btn-stop"  class="rec-btn" onclick="stopRecording()" disabled>■ Stop</button>
        <span   id="rec-status" style="margin-left:10px; color:#e0e6ff; font-size:0.85rem;">Ready</span>
        <div    id="rec-error"  style="color:#ff6961; margin-top:6px; font-size:0.8rem;"></div>

        <script>
        let mediaRecorder;
        let chunks = [];

        function setStatus(msg, isRecording) {
            const el = document.getElementById("rec-status");
            if (!el) return;
            el.innerHTML = isRecording
                ? `<span class="recording-dot"></span>${msg}`
                : msg;
        }
        function setError(msg) {
            const el = document.getElementById("rec-error");
            if (el) el.textContent = msg;
        }
        function setBtns(recording) {
            const s = document.getElementById("btn-start");
            const t = document.getElementById("btn-stop");
            if (s) s.disabled = recording;
            if (t) t.disabled = !recording;
        }

        async function startRecording() {
            setError("");
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                chunks = [];

                mediaRecorder.ondataavailable = (e) => {
                    if (e.data && e.data.size > 0) chunks.push(e.data);
                };

                mediaRecorder.onstop = () => {
                    try {
                        const blob = new Blob(chunks, { type: "audio/webm" });
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            Streamlit.setComponentValue(reader.result);
                            setStatus("Done — processing...");
                            setBtns(false);
                        };
                        reader.onerror = () => {
                            setError("Failed to read recording data.");
                            Streamlit.setComponentValue(null);
                            setBtns(false);
                        };
                        reader.readAsDataURL(blob);
                    } catch (err) {
                        setError("Failed to process recording: " + err.message);
                        Streamlit.setComponentValue(null);
                        setBtns(false);
                    }
                };

                mediaRecorder.start(100);   // collect data every 100 ms
                setStatus("Recording...", true);
                setBtns(true);
            } catch (err) {
                if (err.name === "NotAllowedError") {
                    setError("Microphone access denied. Please allow microphone and try again.");
                } else {
                    setError("Could not start recording: " + err.message);
                }
                Streamlit.setComponentValue(null);
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                // Stop all tracks to release the mic
                mediaRecorder.stream.getTracks().forEach(t => t.stop());
                setStatus("Processing...");
            }
        }
        </script>
        """,
        height=90,
    )

    if recorded is None:
        return None

    # recorded is a data-URL string like "data:audio/webm;base64,..."
    if isinstance(recorded, str) and recorded.startswith("data:"):
        try:
            return _parse_data_url(recorded)
        except Exception as exc:
            st.error(f"Failed to decode recorded audio: {exc}")
            return None

    return None


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

# ── Auto-submit pending voice query ──────────────────────────────────────────
if st.session_state.pending_voice_query:
    pending = st.session_state.pending_voice_query
    st.session_state.pending_voice_query = None   # clear before running
    _run_query(pending)
    st.rerun()

# ── Text input row ────────────────────────────────────────────────────────────
input_col, action_col = st.columns([0.78, 0.22])
prompt = input_col.text_input(
    "Message",
    key="prompt",
    placeholder="Ask anything...",
    label_visibility="collapsed",
)

with action_col:
    send = st.button("Send")
    record_clicked = st.button("🎙️ Record")

if record_clicked:
    st.session_state.recording = True
    st.session_state.recorded_audio = None
    st.session_state.audio_processed = False

if send and prompt:
    _run_query(prompt)

# ── Voice recorder UI ─────────────────────────────────────────────────────────
if st.session_state.recording:
    st.markdown("---")
    st.markdown("**Record your question (click ▶ Start, speak, then ■ Stop):**")

    audio_bytes = _audio_recorder()

    if audio_bytes:
        st.session_state.recorded_audio = audio_bytes
        st.session_state.recorded_audio_format = "webm"
        st.session_state.recording = False
        st.rerun()

# ── Transcription step ────────────────────────────────────────────────────────
if st.session_state.recorded_audio and not st.session_state.audio_processed:

    fmt = st.session_state.recorded_audio_format or "webm"

    with st.spinner("Converting audio..."):
        try:
            wav_path = _convert_to_wav(st.session_state.recorded_audio, fmt)
        except Exception as exc:
            st.error(
                f"Audio conversion failed: {exc}. "
                "Make sure ffmpeg is installed (`sudo apt-get install ffmpeg`)."
            )
            st.session_state.recorded_audio = None
            st.session_state.audio_processed = False
            st.stop()

    with st.spinner("Transcribing audio..."):
        try:
            result, _ = bot.ask_audio(wav_path, lang="auto")
        except Exception as exc:
            st.error(f"Audio pipeline failed: {exc}")
            st.session_state.recorded_audio = None
            st.session_state.audio_processed = False
            st.stop()
        finally:
            try:
                os.remove(wav_path)
            except OSError:
                pass

    transcribed = result.get("query") or ""

    if not transcribed.strip():
        st.warning("No speech detected — please try recording again.")
        st.session_state.recorded_audio = None
        st.session_state.audio_processed = False
        st.stop()

    # Store and auto-submit on next rerun
    st.session_state.audio_processed = True
    st.session_state.recorded_audio = None
    st.session_state.pending_voice_query = transcribed
    st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
with history_container:
    _render_history()