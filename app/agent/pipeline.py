from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import Config, logger
from .ingestion import load_documents, chunk_documents, build_or_load_vectorstore
from .retrieval import build_retriever
from .generation import generation_prompt, format_docs
from .safety import is_safety_critical, verify_safety
from app.agent.stt import Speech2Text
from app.agent.tts import Text2Speech


ABSTENTION_MESSAGE = (
    "⚠️  This question requires verification from an extension officer. "
    "The information I retrieved does not fully support a safe answer. "
    "Please contact your local FMARD extension office or call the Nigeria "
    "Agriculture Hotline."
)

NO_CONTEXT_MESSAGE = (
    "I don't have that information in my knowledge base. "
    "Please consult your local agricultural extension officer."
)

# Per-language voice mapping for Spitch TTS
# "sade" is English-only; use language-appropriate voices for others
LANG_VOICE_MAP = {
    "en": "sade",
    "yo": "tunde",   # Yoruba voice — update to actual Spitch voice name for yo
    "ha": "tunde",   # Hausa
    "ig": "tunde",   # Igbo
    "pcm": "tunde",  # Nigerian Pidgin
}
DEFAULT_VOICE = "tunde"


class NaijaAgroChat:
    """
    Full NaijaAgroChat pipeline.

    Usage:
        bot = NaijaAgroChat.build()
        response = bot.ask("My cassava get disease, wetin I go do?")
        print(response["answer"])
    """

    def __init__(self, retriever, generator_llm, safety_llm: ChatOpenAI):
        self.retriever = retriever
        self.generator_llm = generator_llm
        self.safety_llm = safety_llm

    # ── public API ─────────────────────────────────────────────────────────────

    def ask(self, query: str) -> dict:
        """
        Process a farmer query end-to-end.

        Returns:
            {
                "query": str,
                "answer": str,
                "sources": List[str],
                "safety_checked": bool,
                "safe": bool | None,
            }
        """
        logger.info(f"Query: {query}")

        # 1. Retrieve context ONCE
        retrieved_docs = self.retriever.invoke(query)
        context_str = format_docs(retrieved_docs)
        sources = list({d.metadata.get("source", "unknown") for d in retrieved_docs})

        if not retrieved_docs:
            logger.warning("No documents retrieved — abstaining.")
            return {
                "query": query,
                "answer": NO_CONTEXT_MESSAGE,
                "sources": [],
                "safety_checked": False,
                "safe": None,
            }

        # 2. Generate — pass pre-retrieved context directly, no second retrieval
        try:
            answer = (
                generation_prompt
                | self.generator_llm
                | StrOutputParser()
            ).invoke({"question": query, "context": context_str})
        except Exception as e:
            logger.error(f"Generation failed: {e}")
            raise

        # 3. Safety check (only for safety-critical queries)
        safety_checked = False
        is_safe: Optional[bool] = None

        if is_safety_critical(query):
            safety_checked = True
            is_safe, reason = verify_safety(query, answer, context_str, self.safety_llm)
            logger.info(f"Safety check — safe={is_safe}, reason={reason}")
            if not is_safe:
                answer = ABSTENTION_MESSAGE

        return {
            "query": query,
            "answer": answer,
            "sources": sources,
            "safety_checked": safety_checked,
            "safe": is_safe,
        }

    # ── audio helpers ─────────────────────────────────────────────────────────

    def ask_audio(self, audio_file: str, lang: str = "en") -> Tuple[dict, object]:
        """Transcribe audio → RAG query → TTS response.

        Returns (result_dict, audio_segment).
        NOTE: STT, RAG, and TTS are deliberately left as separate steps so
        the Streamlit frontend can wrap each in its own spinner/error block.
        """
        text = Speech2Text(audio_file, lang)
        result = self.ask(text)
        audio = Text2Speech(result["answer"], lang)
        return result, audio

    def text_to_speech(self, text: str, lang: str = "en"):
        """Generate speech from plain text."""
        return Text2Speech(text, lang)

    # ── factory ───────────────────────────────────────────────────────────────

    @classmethod
    def build(
        cls,
        docs_dir: Optional[str] = None,
        force_rebuild_index: bool = False,
        openai_api_key: Optional[str] = None,
        use_openai_generator: bool = False,
    ) -> "NaijaAgroChat":

        docs_dir = docs_dir or Config.DOCS_DIR

        # ── Vector store ──────────────────────────────────────────────────────
        index_path = Path(Config.FAISS_INDEX_DIR)
        if index_path.exists() and not force_rebuild_index:
            vectorstore = build_or_load_vectorstore()
        else:
            documents = load_documents(docs_dir)
            chunks = chunk_documents(documents) if documents else []
            if not chunks:
                logger.warning(
                    "No documents found. Building an empty index. "
                    "Add PDFs to '%s' and rebuild.", docs_dir,
                )
                from langchain_core.documents import Document as _Doc
                chunks = [_Doc(
                    page_content="No agricultural documents loaded yet.",
                    metadata={"source": "placeholder"},
                )]
            vectorstore = build_or_load_vectorstore(chunks, force_rebuild=True)

        # ── Retriever ─────────────────────────────────────────────────────────
        retriever = build_retriever(vectorstore)

        # ── Generator ─────────────────────────────────────────────────────────
        from app.config.settings import dot_env_pth
        import os

        # Resolve key once — never pass None to ChatOpenAI
        # (passing None triggers an async callable bug in langchain-openai)
        os.environ["OPENAI_API_KEY"] = dot_env_pth.OPENAI_API_KEY

        if use_openai_generator:
            logger.info("Using OpenAI GPT-4o-mini as generator.")
            generator_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=dot_env_pth.OPENAI_API_KEY,
            )
        else:
            logger.info(f"Loading local generator: {Config.GENERATOR_MODEL}")
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            from transformers import logging as hf_logging
            hf_logging.set_verbosity_error()
            import torch

            tokenizer = AutoTokenizer.from_pretrained(
                Config.GENERATOR_MODEL, token=dot_env_pth.HUGGINGFACE_API_KEY
            )
            model = AutoModelForCausalLM.from_pretrained(
                Config.GENERATOR_MODEL,
                token=dot_env_pth.HUGGINGFACE_API_KEY,
                torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                device_map="auto",
            )
            pipe = pipeline(
                "text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=Config.MAX_NEW_TOKENS,
                max_length=Config.MAX_TOTAL_TOKENS,
                do_sample=False,
            )
            from langchain_huggingface import HuggingFacePipeline
            generator_llm = HuggingFacePipeline(pipeline=pipe)

        # ── Safety LLM ────────────────────────────────────────────────────────
        safety_llm = ChatOpenAI(
            model=Config.OPENAI_SAFETY_MODEL,
            temperature=0,
            api_key=dot_env_pth.OPENAI_API_KEY,
        )

        logger.info("NaijaAgroChat pipeline ready.")
        return cls(retriever, generator_llm, safety_llm)