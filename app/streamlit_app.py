import streamlit as st
from pathlib import Path
import tempfile
from io import BytesIO

from app.agent import NaijaAgroChat
from app.agent.stt import Speech2Text

st.set_page_config(page_title="NaijaAgroChat", layout="wide")


@st.cache_resource
def get_bot():
    # use_openai_generator=True avoids loading the 8B Aya model locally.
    # The local model requires ~16GB RAM and crashes via disk-offload on
    # consumer hardware. GPT-4o-mini handles generation instead.
    return NaijaAgroChat.build(use_openai_generator=True)


bot = get_bot()

st.title("NaijaAgroChat 🔆")

mode = st.radio("Input mode", ["Text", "Audio"])

# ── Text mode ─────────────────────────────────────────────────────────────────
if mode == "Text":
    question = st.text_area("Ask a question", height=100)
    if st.button("Submit") and question.strip():

        with st.spinner("Retrieving & generating answer..."):
            try:
                result = bot.ask(question)
            except Exception as exc:
                st.error(f"Pipeline error: {exc}")
                st.stop()

        st.markdown("**Answer:**")
        st.write(result.get("answer", "<no answer returned>"))
        if result.get("sources"):
            st.markdown("**Sources:**")
            st.write(result["sources"])
        if result.get("safe") is not None:
            st.markdown("**Safety check:**")
            st.write(result["safe"])

        with st.spinner("Generating audio response..."):
            try:
                audio_seg = bot.text_to_speech(result.get("answer", ""), lang="en")
                buf = BytesIO()
                audio_seg.export(buf, format="mp3")
                st.audio(buf.getvalue(), format="audio/mp3")
            except Exception as exc:
                st.warning(f"TTS failed (answer still shown above): {exc}")

# ── Audio mode ────────────────────────────────────────────────────────────────
else:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    lang = st.text_input("Language code", value="en")

    if uploaded and st.button("Transcribe & Ask"):

        # Write upload to a temp file once
        with tempfile.NamedTemporaryFile(
            delete=False, suffix=Path(uploaded.name).suffix
        ) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        # ── Stage 1: Transcription ────────────────────────────────────────────
        with st.spinner("Transcribing audio..."):
            try:
                transcribed_text = Speech2Text(tmp_path, lang)
            except Exception as exc:
                st.error(f"Transcription failed: {exc}")
                st.stop()

        st.markdown("**Transcribed query:**")
        st.write(transcribed_text)   # visible immediately — confirms STT worked

        # ── Stage 2: RAG pipeline ─────────────────────────────────────────────
        with st.spinner("Retrieving & generating answer..."):
            try:
                result = bot.ask(transcribed_text)
            except Exception as exc:
                st.error(f"RAG pipeline failed: {exc}")
                st.stop()

        st.markdown("**Answer:**")
        st.write(result["answer"])
        if result.get("sources"):
            st.markdown("**Sources:**")
            st.write(result["sources"])
        if result.get("safe") is not None:
            st.markdown("**Safety check:**")
            st.write(result["safe"])

        # ── Stage 3: TTS ──────────────────────────────────────────────────────
        with st.spinner("Generating audio response..."):
            try:
                audio_seg = bot.text_to_speech(result["answer"], lang=lang)
                buf = BytesIO()
                audio_seg.export(buf, format="mp3")
                st.audio(buf.getvalue(), format="audio/mp3")
            except Exception as exc:
                st.warning(f"TTS failed (answer still shown above): {exc}")