import base64
import os
import streamlit as st
from io import BytesIO
import tempfile

from app.agent import NaijaAgroChat

try:
    from aiortc.contrib.media import MediaRecorder
    from streamlit_webrtc import WebRtcMode, webrtc_streamer

    HAS_WEBRTC = True
except ImportError:
    HAS_WEBRTC = False

st.set_page_config(page_title="NaijaAgroChat", layout="wide")


@st.cache_resource
def get_bot():
    return NaijaAgroChat.build(use_openai_generator=True)


bot = get_bot()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "recording" not in st.session_state:
    st.session_state.recording = False
if "recorded_audio" not in st.session_state:
    st.session_state.recorded_audio = None
if "audio_processed" not in st.session_state:
    st.session_state.audio_processed = False
if "audio_transcription" not in st.session_state:
    st.session_state.audio_transcription = ""
if "recorded_audio_format" not in st.session_state:
    st.session_state.recorded_audio_format = "webm"
# Holds a voice query that should be auto-submitted on the next run
if "pending_voice_query" not in st.session_state:
    st.session_state.pending_voice_query = None


def _append_message(role: str, content: str):
    st.session_state.chat_history.append({"role": role, "content": content})


def _render_history():
    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])


def _parse_data_url(data_url: str) -> bytes:
    _, encoded = data_url.split(",", 1)
    return base64.b64decode(encoded)


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


def _audio_recorder() -> bytes | None:
    """In-browser recorder that returns WebM bytes."""

    recorded = st.components.v1.html(
        """
        <style>
            .recording-indicator {
                display: inline-block;
                width: 0.7rem;
                height: 0.7rem;
                border-radius: 50%;
                background: #ff4b4b;
                margin-right: 0.45rem;
                vertical-align: middle;
                animation: pulse 1.2s ease-in-out infinite;
            }

            @keyframes pulse {
                0% { transform: scale(1); opacity: 0.9; }
                50% { transform: scale(1.35); opacity: 0.5; }
                100% { transform: scale(1); opacity: 0.9; }
            }
        </style>
        <script>
        let mediaRecorder;
        let chunks = [];

        function setStatus(msg, isRecording = false) {
            const statusEl = document.getElementById("record-status");
            if (!statusEl) return;
            if (isRecording) {
                statusEl.innerHTML = `<span class="recording-indicator"></span>${msg}`;
            } else {
                statusEl.textContent = msg;
            }
        }

        function setError(msg) {
            const errEl = document.getElementById("record-error");
            if (errEl) errEl.textContent = msg;
        }

        async function startRecording() {
            setError("");
            try {
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                mediaRecorder = new MediaRecorder(stream);
                chunks = [];

                mediaRecorder.ondataavailable = (e) => chunks.push(e.data);
                mediaRecorder.onstop = async () => {
                    try {
                        const blob = new Blob(chunks, { type: "audio/webm" });
                        const reader = new FileReader();
                        reader.onloadend = () => {
                            Streamlit.setComponentValue(reader.result);
                            setStatus("Done — submitting...");
                        };
                        reader.onerror = () => {
                            setError("Failed to read recording data.");
                            Streamlit.setComponentValue({ error: "read_error" });
                        };
                        reader.readAsDataURL(blob);
                    } catch (err) {
                        setError("Failed to process recording.");
                        Streamlit.setComponentValue({ error: "process_error" });
                    }
                };

                mediaRecorder.start();
                setStatus("Recording...", true);
            } catch (err) {
                setError("Microphone access denied.");
                Streamlit.setComponentValue({ error: "permission_denied" });
            }
        }

        function stopRecording() {
            if (mediaRecorder && mediaRecorder.state !== "inactive") {
                mediaRecorder.stop();
                setStatus("Processing...");
            }
        }
        </script>

        <button onclick="startRecording()">Start recording</button>
        <button onclick="stopRecording()">Stop recording</button>
        <span id="record-status" style="margin-left:10px;color:#e0e6ff;">Ready</span>
        <div id="record-error" style="color:#ff6961;margin-top:8px;"></div>
        """,
        height=80,
    )

    if recorded:
        if isinstance(recorded, dict) and recorded.get("error"):
            st.error(f"Audio recorder error: {recorded['error']}")
            return None
        try:
            return _parse_data_url(recorded)
        except Exception as exc:
            st.error(f"Failed to decode recorded audio: {exc}")
            return None
    return None


def _webrtc_recorder() -> bytes | None:
    """Record audio via streamlit-webrtc and return raw bytes."""

    if not HAS_WEBRTC:
        st.info("Install `streamlit-webrtc` to use the improved recorder. Falling back to the built-in recorder.")
        return _audio_recorder()

    # Create a temp file for the recorded audio; reuse across reruns until consumed.
    if "webrtc_tmpfile" not in st.session_state or not st.session_state.webrtc_tmpfile:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        st.session_state.webrtc_tmpfile = tmp.name
        tmp.close()

    tmp_path = st.session_state.webrtc_tmpfile

    def _recorder_factory():
        return MediaRecorder(tmp_path, format="wav")

    RTC_CONFIGURATION = {
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},          # STUN server
        {"urls": ["turn:YOUR_TURN_SERVER:3478"],            # TURN server
         "username": "user",
         "credential": "pass"}
    ]
}
    ctx = webrtc_streamer(
    key="audio_recorder",
    mode=WebRtcMode.SENDONLY,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"audio": True, "video": False},
    in_recorder_factory=_recorder_factory,
    audio_receiver_size=256,
)

    # If the stream is still playing, wait for it to stop.
    if ctx.state.playing:
        st.info("Recording... When you stop, your audio will be processed.")
        return None

    # The recording has stopped; return the captured file contents.
    if os.path.exists(tmp_path) and os.path.getsize(tmp_path) > 0:
        with open(tmp_path, "rb") as f:
            data = f.read()

        # Clean up so subsequent recordings create a new file.
        st.session_state.webrtc_tmpfile = None
        try:
            os.remove(tmp_path)
        except OSError:
            pass

        return data

    return None


# ── Page layout ──────────────────────────────────────────────────────────────

st.title("NaijaAgroChat 🔆")

st.markdown(
    """
    <style>
    body, .stApp { background: #0b0e14 !important; color: #e0e6ff !important; }
    .stTextInput > div > input {
        background: #0f1220 !important; color: #e0e6ff !important;
        border: 1px solid #2e3a58 !important;
    }
    .stButton button {
        background: #2563eb !important; color: white !important;
        border-radius: 0.75rem !important; height: 2.4rem !important;
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
    st.session_state.pending_voice_query = None   # clear before running
    _run_query(pending)
    st.rerun()  # re-render so the new messages show up cleanly

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
    record_clicked = st.button("🎙️")

if record_clicked:
    st.session_state.recording = True
    st.session_state.recorded_audio = None
    st.session_state.audio_processed = False

if send and prompt:
    _run_query(prompt)

# ── Voice recorder UI ─────────────────────────────────────────────────────────
if st.session_state.recording:
    st.markdown("---")
    st.markdown("**Record a question (voice input):**")
    audio_bytes = _webrtc_recorder()

    if audio_bytes:
        st.session_state.recorded_audio = audio_bytes
        st.session_state.recorded_audio_format = "wav" if HAS_WEBRTC else "webm"
        st.session_state.recording = False
        st.rerun()  # move on to transcription step

# ── Transcription step ────────────────────────────────────────────────────────
if st.session_state.recorded_audio and not st.session_state.audio_processed:
    suffix = f".{st.session_state.recorded_audio_format or 'webm'}"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(st.session_state.recorded_audio)
        tmp_path = tmp.name

    with st.spinner("Transcribing audio..."):
        try:
            result, _ = bot.ask_audio(tmp_path, lang="auto")
        except Exception as exc:
            st.error(f"Audio pipeline failed: {exc}")
            # Reset so the user can try again
            st.session_state.recorded_audio = None
            st.session_state.audio_processed = False
            st.stop()

    transcribed = result.get("query") or ""

    # Mark as processed and store query; the next rerun will auto-submit it.
    st.session_state.audio_processed = True
    st.session_state.recorded_audio = None
    st.session_state.pending_voice_query = transcribed
    st.rerun()

# ── Render chat history ───────────────────────────────────────────────────────
with history_container:
    _render_history()