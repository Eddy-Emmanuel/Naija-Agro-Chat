# Naija‑Agro‑Chat

An AI‑powered conversational assistant tailored for Nigerian agriculture.  
This project integrates speech‑to‑text, text generation, retrieval over domain documents, and text‑to‑speech to deliver an interactive experience via a Streamlit frontend.

---

## 📌 Features

- **Speech‑to‑Text (STT)** for capturing user queries verbally.
- **Text Generation** using an LLM to answer agricultural questions.
- **Retrieval Pipeline** that searches a FAISS index built from domain knowledge.
- **Safety checks and moderation** to filter inappropriate inputs.
- **Text‑to‑Speech (TTS)** to read responses aloud.
- **Streamlit UI** for web‑based interaction.

---

## 📁 Repository Structure

```
.
├── app/
│   ├── streamlit_app.py          # entry point for the Streamlit front end
│   ├── agent/                    # core AI “agent” modules
│   │   ├── config.py             # configuration/constants
│   │   ├── generation.py         # LLM prompt building & text generation
│   │   ├── ingestion.py          # document ingestion helpers
│   │   ├── pipeline.py           # orchestration of retrieval & generation
│   │   ├── retrieval.py          # FAISS index search logic
│   │   ├── safety.py             # input/output safety checks
│   │   ├── stt.py                # speech‑to‑text helper
│   │   ├── tts.py                # text‑to‑speech helper
│   │   └── __init__.py
│   └── agent/…                    # other supporting modules
├── config/
│   └── settings.py               # environment/configuration settings
├── docs/                         # additional documentation
├── faiss_index/
│   └── index.faiss               # prebuilt FAISS vector index
├── requirements.txt              # Python dependencies
└── README.md                     # this file
```

---

## ⚙️ Setup & Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/your_org/NaijaAgroChat.git
   cd NaijaAgroChat
   ```

2. **Create a Python environment**

   ```bash
   python -m venv .venv
   .venv\Scripts\activate        # Windows
   source .venv/bin/activate     # macOS/Linux
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Configuration**

   - Edit `config/settings.py` to supply API keys (e.g. OpenAI, short‑term STT/TTS credentials) and other environment variables.
   - Adjust model names, FAISS path, or any constants inside `app/agent/config.py` as needed.

5. **Prepare data (optional)**

   Run any ingestion scripts in `app/agent/ingestion.py` to build or update the FAISS index from your own documents; the prebuilt index lives in `faiss_index/index.faiss`.

6. **Run the app**

   ```bash
   streamlit run app/streamlit_app.py
   ```

   The UI will be served locally (usually at `http://localhost:8501`).

---

## 🚀 Usage

- **Web UI** – Type or speak questions about Nigerian agriculture; the agent responds in text and voice.
- **Programmatic** – Import components from `app.agent` to build custom workflows or CLI tools.

---

## 🔍 Architecture Overview

1. **User Input**  
   - Streamlit captures text or audio.  
   - Audio is converted via `stt.py`.

2. **Safety Check**  
   - `safety.py` inspects input for harmful content.

3. **Retrieval**  
   - `retrieval.py` searches the FAISS index for relevant passages.

4. **Generation**  
   - `generation.py` formats prompts combining query + retrieved context.  
   - Sends to LLM (e.g. OpenAI GPT) and receives response.

5. **Post‑processing & Safety**  
   - Output is filtered again before being shown.

6. **Text‑to‑Speech**  
   - `tts.py` converts the textual answer to audio.

7. **Frontend**  
   - `streamlit_app.py` glues together the above modules into an interactive experience.

---

## 🛠 Development

- **Adding new data**: Update ingestion routines and rebuild FAISS index.
- **Changing models**: Modify `config.MODEL_NAME` or related constants.
- **Extending functionality**: New pipelines or commands belong under `app/agent/`.

Helpful commands:

```bash
# linting / formatting
flake8 app tests
black .

# run tests (add tests under a tests/ directory)
pytest
```

---

## 🤝 Contributing

1. Fork the repo and create a branch.
2. Ensure code follows style guidelines and includes tests.
3. Open a pull request describing your changes.
4. Be sure to update this README if behavior or configuration changes.

---

## 📄 License

Specify your license here (e.g. MIT, Apache‑2.0).

---

## 📝 Acknowledgments

- Built with [Streamlit](https://streamlit.io/)
- Uses [FAISS](https://github.com/facebookresearch/faiss) for vector similarity search
- Powered by [OpenAI](https://openai.com/) or whichever backend you configure

---

> Let me know if you’d like a condensed README for end‑users, or additional developer docs in `docs/`.