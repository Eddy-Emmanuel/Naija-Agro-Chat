from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

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
    "yo": "tunde",   # Yoruba voice
    "ha": "tunde",   # Hausa
    "ig": "tunde",   # Igbo
    "pcm": "tunde",  # Nigerian Pidgin
}
DEFAULT_VOICE = "tunde"

# ── Agent / tool imports ──────────────────────────────────────────────────────
from langchain_core.tools import Tool                                  
from langchain_classic.agents.agent_types import AgentType
from langchain_classic.agents.initialize import initialize_agent


class NaijaAgroChat:
    """Full NaijaAgroChat pipeline.

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
        agent_executor=None,
    ):
        self.retriever = retriever
        self.generator_llm = generator_llm
        self.safety_llm = safety_llm
        self.translator_llm = translator_llm
        self.agent_executor = agent_executor
        self._last_tool_output: str = ""

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
                "lang": str,
            }
        """
        logger.info(f"Query: {query}")
        detected_lang = detect_language(query)

        # Include the current date so the model can reason about time-sensitive matters.
        current_date = datetime.now().astimezone().isoformat()

        # 0. First try to answer directly (without invoking retrieval/web tools) so we
        #    don't call external tools for questions the model can answer on its own.
        direct_answer = self._try_answer_without_tools(query, current_date)
        if direct_answer is not None:
            answer = direct_answer
            sources = []
            context_for_safety = ""
            # Safety check still applies if the query is safety-critical.
            safety_checked = False
            is_safe: Optional[bool] = None
            if is_safety_critical(query):
                safety_checked = True
                is_safe, reason = verify_safety(query, answer, context_for_safety, self.safety_llm)
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

        # 1. Translate non-English queries to English for retrieval
        #    (documents are primarily English)
        query_for_retrieval, translated = self._translate_query_for_retrieval(query)

        # 2. Retrieve context ONCE
        retrieved_docs = self.retriever.invoke(query_for_retrieval)
        context_str = format_docs(retrieved_docs)
        sources = list({d.metadata.get("source", "unknown") for d in retrieved_docs})

        # If the index is empty, the build step may insert a placeholder doc.
        # Treat that as "no context" so web search can act as a fallback.
        no_docs = (
            not retrieved_docs
            or all(d.metadata.get("source") == "placeholder" for d in retrieved_docs)
            or ("No agricultural documents" in context_str)
        )

        if no_docs:
            logger.warning("No documents retrieved — trying web search fallback.")

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
                # Still no useful context — respond with a safe fallback message.
                message = NO_CONTEXT_MESSAGE
                if translated:
                    message = self._localize_message_for_query(message, query)
                return {
                    "query": query,
                    "answer": message,
                    "sources": [],
                    "safety_checked": False,
                    "safe": None,
                    "lang": detected_lang,
                }

        # If we have an agent_executor (tool-based agent), let it handle
        # retrieval + generation.
        if self.agent_executor:
            logger.info("Using agent executor for query.")
            # Include the current date so the model can reason about time-sensitive matters.
            answer = self.agent_executor.run(
                f"Current date: {current_date}\n\n{query}"
            )
            context_for_safety = ""  # agent tools are opaque here
        else:
            # 2. Generate — pass pre-retrieved context directly
            try:
                answer = (
                    generation_prompt
                    | self.generator_llm
                    | StrOutputParser()
                ).invoke(
                    {
                        "question": query,
                        "context": context_str,
                        "current_date": current_date,
                    }
                )
            except Exception as e:
                logger.error(f"Generation failed: {e}")
                raise
            context_for_safety = context_str

        # 3. Safety check (only for safety-critical queries)
        safety_checked = False
        is_safe: Optional[bool] = None

        if is_safety_critical(query):
            safety_checked = True
            is_safe, reason = verify_safety(query, answer, context_for_safety, self.safety_llm)
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

    def _try_answer_without_tools(self, query: str, current_date: str) -> Optional[str]:
        """Try to answer without retrieving context or calling web tools.

        Returns a non-empty answer only if the model appears confident.
        """
        try:
            answer = (
                generation_prompt
                | self.generator_llm
                | StrOutputParser()
            ).invoke({
                "question": query,
                "context": "",
                "current_date": current_date,
            })
        except Exception as e:
            logger.warning("Direct answer attempt failed: %s", e)
            return None

        if not answer:
            return None

        lower = answer.strip().lower()
        # If the model indicates it can't answer without context, treat as unknown.
        for phrase in [
            "i don't have",
            "i dont have",
            "i don't know",
            "i dont know",
            "i'm not sure",
            "im not sure",
            "no information",
            "no relevant",
            "i cannot",
            "i cant",
            "i can't",
            "i don't have enough",
        ]:
            if phrase in lower:
                return None

        return answer

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

    def ask_audio(self, audio_file: str, lang: str = "auto") -> Tuple[dict, None]:
        """Transcribe audio → RAG query.

        Returns (result_dict, None).
        TTS is handled separately by the caller (app.py) via text_to_speech().
        STT, RAG, and TTS are kept as separate steps so the Streamlit frontend
        can wrap each in its own spinner/error block.
        """
        # Spitch requires a language code; if user requested auto-detection,
        # transcribe with English and then detect the actual language from text.
        stt_lang = "en" if lang == "auto" else lang
        text = Speech2Text(audio_file, stt_lang)
        result = self.ask(text)
        # TTS is intentionally NOT called here — app.py calls text_to_speech()
        # separately to avoid a double TTS round-trip.
        return result, None

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
        use_agent: bool = False,
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
        os.environ["OPENAI_API_KEY"] = dot_env_pth.OPENAI_API_KEY

        logger.info("Using OpenAI GPT-4o-mini as generator.")
        generator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0.1,
            api_key=dot_env_pth.OPENAI_API_KEY,
        )

        # ── Safety LLM ────────────────────────────────────────────────────────
        safety_llm = ChatOpenAI(
            model=Config.OPENAI_SAFETY_MODEL,
            temperature=0,
            api_key=dot_env_pth.OPENAI_API_KEY,
        )

        # ── Translator LLM ────────────────────────────────────────────────────
        translator_llm = ChatOpenAI(
            model="gpt-4o-mini",
            temperature=0,
            api_key=dot_env_pth.OPENAI_API_KEY,
        )

        # ── Optional agent (retriever + web search tools) ─────────────────────
        agent_executor = None
        if use_agent:
            # Tools return plain text summaries consumed by the ReAct agent.
            def _kb_tool(q: str) -> str:
                docs = retriever.invoke(q)           # use .invoke(), not deprecated .get_relevant_documents()
                return format_docs(docs) or "No relevant documents found."

            def _web_tool(q: str) -> str:
                try:
                    results = web_search(q, num_results=Config.WEB_SEARCH_RESULTS)
                    return format_search_results(results) or "No web results found."
                except Exception as e:
                    return f"Web search failed: {e}"

            tools = [
                Tool(
                    name="knowledge_base",
                    func=_kb_tool,
                    description=(
                        "Search the local agricultural knowledge base. "
                        "Return a concise summary of the most relevant passages."
                    ),
                ),
                Tool(
                    name="web_search",
                    func=_web_tool,
                    description=(
                        "Search the web for recent information. "
                        "Return a concise summary of search results."
                    ),
                ),
            ]

            agent_executor = initialize_agent(
                tools,
                generator_llm,
                agent=AgentType.CHAT_ZERO_SHOT_REACT_DESCRIPTION,
                verbose=True,
                # max_iterations=3,
                handle_parsing_errors=True,
            )

        logger.info("NaijaAgroChat pipeline ready.")
        return cls(
            retriever,
            generator_llm,
            safety_llm,
            translator_llm,
            agent_executor=agent_executor,
        )