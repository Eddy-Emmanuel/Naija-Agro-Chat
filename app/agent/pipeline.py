from pathlib import Path
from typing import List, Optional, Tuple

from langdetect import DetectorFactory, detect
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import Config, logger
from .ingestion import load_documents, chunk_documents, build_or_load_vectorstore
from .retrieval import build_retriever
from .generation import (
    generation_prompt,
    format_docs,
    translation_prompt,
    translate_to_query_lang_prompt,
)
from .safety import is_safety_critical, verify_safety
from .web_search import format_search_results, web_search
from app.agent.stt import Speech2Text
from app.agent.tts import Text2Speech

# langdetect is non-deterministic by default; seed it for stable results
DetectorFactory.seed = 0

SUPPORTED_LANGS = {"en", "yo", "ha", "ig", "pcm"}


def detect_language(text: str) -> str:
    """Naively detect the user language from a text string.

    Falls back to English if detection fails or returns an unsupported code.
    """

    try:
        lang = detect(text)
    except Exception:
        return "en"

    if lang in SUPPORTED_LANGS:
        return lang
    # Some detections may return 'pt' etc. When unsure, fall back to English.
    return "en"


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

    def __init__(
        self,
        retriever,
        generator_llm,
        safety_llm: ChatOpenAI,
        translator_llm: ChatOpenAI,
    ):
        self.retriever = retriever
        self.generator_llm = generator_llm
        self.safety_llm = safety_llm
        self.translator_llm = translator_llm

    # ── public API ─────────────────────────────────────────────────────────────

    def ask(self, query: str) -> dict:
        """Process a farmer query end-to-end.

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
        detected_lang = detect_language(query)

        # 0. Translate non-English queries to English for retrieval (documents are primarily English)
        query_for_retrieval, translated = self._translate_query_for_retrieval(query)

        # 1. Retrieve context ONCE
        retrieved_docs = self.retriever.invoke(query_for_retrieval)
        context_str = format_docs(retrieved_docs)
        sources = list({d.metadata.get("source", "unknown") for d in retrieved_docs})

        if not retrieved_docs:
            logger.warning("No documents retrieved — trying web search fallback.")

            # If the knowledge base has nothing, optionally fall back to live web search.
            if Config.USE_WEB_SEARCH:
                try:
                    web_results = web_search(
                        query_for_retrieval, num_results=Config.WEB_SEARCH_RESULTS
                    )
                except Exception as e:
                    logger.warning(f"Web search failed: {e}")
                    web_results = []

                if web_results:
                    context_str = format_search_results(web_results)
                    sources = [r.get("link", "unknown") for r in web_results]

            if not context_str:
                # Still no useful context – respond with a safe fallback message.
                message = NO_CONTEXT_MESSAGE
                if translated:
                    # Return the fallback message in the same language as the query.
                    message = self._localize_message_for_query(message, query)
                return {
                    "query": query,
                    "answer": message,
                    "sources": [],
                    "safety_checked": False,
                    "safe": None,
                    "lang": detected_lang,
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
            "lang": detected_lang,
        }

    def _translate_query_for_retrieval(self, query: str) -> tuple[str, bool]:
        """Translate the query into English for better retrieval quality.

        Documents in the knowledge base are primarily English, so translating
        non-English queries improves RAG retrieval.

        Returns:
            (translated_query, was_translation_performed)
        """
        try:
            translation = (
                translation_prompt
                | self.translator_llm
                | StrOutputParser()
            ).invoke({"text": query})
            translation = translation.strip()
            translated = bool(translation and translation.strip() != query.strip())
            return (translation or query, translated)
        except Exception as e:
            logger.warning(
                "Query translation failed (using original query): %s", e
            )
            return query, False

    def _localize_message_for_query(self, message: str, query: str) -> str:
        """Translate a fallback message into the language of the original query."""
        try:
            localized = (
                translate_to_query_lang_prompt
                | self.translator_llm
                | StrOutputParser()
            ).invoke({"query": query, "message": message})
            return localized.strip() or message
        except Exception as e:
            logger.warning(
                "Message localization failed (using original message): %s", e
            )
            return message

    # ── audio helpers ─────────────────────────────────────────────────────────

    def ask_audio(self, audio_file: str, lang: str = "auto") -> Tuple[dict, object]:
        """Transcribe audio → RAG query → TTS response.

        Returns (result_dict, audio_segment).
        NOTE: STT, RAG, and TTS are deliberately left as separate steps so
        the Streamlit frontend can wrap each in its own spinner/error block.
        """
        # Spitch requires a language code; if user requested auto-detection,
        # transcribe with English and then detect the actual language from text.
        stt_lang = "en" if lang == "auto" else lang
        text = Speech2Text(audio_file, stt_lang)
        result = self.ask(text)

        # If we autodetected, use detected language for TTS
        tts_lang = result.get("lang", lang)
        audio = Text2Speech(result["answer"], tts_lang)
        return result, audio

    def text_to_speech(self, text: str, lang: str = "auto"):
        """Generate speech from plain text."""
        if lang == "auto":
            lang = detect_language(text)
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

        # ── Translator LLM (for query/message localization) ─────────────────────
        translator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=dot_env_pth.OPENAI_API_KEY,
        )

        logger.info("NaijaAgroChat pipeline ready.")
        return cls(retriever, generator_llm, safety_llm, translator_llm)