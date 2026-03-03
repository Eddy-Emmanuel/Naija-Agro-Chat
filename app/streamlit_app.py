import streamlit as st
from pathlib import Path
import tempfile
from io import BytesIO

from app.agent import NaijaAgroChat

# Ensure environment is configured (pydantic settings read .env automatically)

st.set_page_config(page_title="NaijaAgroChat", layout="wide")

@st.cache_resource
def get_bot():
    return NaijaAgroChat.build()

bot = get_bot()

st.title("NaijaAgroChat 🔆")

mode = st.radio("Input mode", ["Text", "Audio"])

if mode == "Text":
    question = st.text_area("Ask a question", height=100)
    if st.button("Submit") and question.strip():
        with st.spinner("Thinking..."):
            result = bot.ask(question)
        st.markdown("**Answer:**")
        st.write(result["answer"])
        if result.get("sources"):
            st.markdown("**Sources:**")
            st.write(result["sources"])
        if result.get("safe") is not None:
            st.markdown("**Safety check:**")
            st.write(result["safe"])

        # produce and play audio
        audio_seg = bot.text_to_speech(result["answer"], lang="en")
        buf = BytesIO()
        audio_seg.export(buf, format="mp3")
        st.audio(buf.getvalue(), format="audio/mp3")

else:
    uploaded = st.file_uploader("Upload audio file", type=["wav", "mp3", "ogg"])
    lang = st.text_input("Language code", value="en")
    if uploaded and st.button("Transcribe & Ask"):
        with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded.name).suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name
        with st.spinner("Transcribing..."):
            result, audio_seg = bot.ask_audio(tmp_path, lang)
        st.markdown("**Transcribed query:**")
        st.write(result["query"])
        st.markdown("**Answer:**")
        st.write(result["answer"])
        if result.get("sources"):
            st.markdown("**Sources:**")
            st.write(result["sources"])
        if result.get("safe") is not None:
            st.markdown("**Safety check:**")
            st.write(result["safe"])

        buf = BytesIO()
        audio_seg.export(buf, format="mp3")
        st.audio(buf.getvalue(), format="audio/mp3")
