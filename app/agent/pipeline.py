from pathlib import Path
from typing import List, Optional, Tuple

from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_openai import ChatOpenAI

from .config import Config, logger
from .ingestion import load_documents, chunk_documents, build_or_load_vectorstore
from .retrieval import build_retriever
from .generation import generation_prompt, format_docs
from .safety import is_safety_critical, verify_safety
from app.agent.stt import Speech2Text
from app.agent.tts import Text2Speech


# messages for abstention / no context
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
    ):
        self.retriever = retriever
        self.generator_llm = generator_llm
        self.safety_llm = safety_llm

        # Core RAG chain (LCEL)
        self._rag_chain = (
            RunnablePassthrough.assign(
                context=RunnableLambda(
                    lambda x: format_docs(self.retriever.invoke(x["question"]))
                )
            )
            | generation_prompt
            | self.generator_llm
            | StrOutputParser()
        )

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

        # 1. Retrieve context
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

        # 2. Generate answer
        answer = self._rag_chain.invoke({"question": query})

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

    # ── convenience helpers for audio ----------------------------------------

    def ask_audio(self, audio_file: str, lang: str = "en") -> Tuple[dict, object]:
        """Transcribe audio, run a query, and produce speech output.

        Returns a tuple of (result_dict, audio_segment).
        """
        text = Speech2Text(audio_file, lang)
        result = self.ask(text)
        audio = Text2Speech(result["answer"], lang)
        return result, audio

    def text_to_speech(self, text: str, lang: str = "en"):
        """Generate speech from plain text."""
        return Text2Speech(text, lang)

    # ── factory --------------------------------------------------------------

    @classmethod
    def build(
        cls,
        docs_dir: Optional[str] = None,
        force_rebuild_index: bool = False,
        openai_api_key: Optional[str] = None,
        use_openai_generator: bool = False,
    ) -> "NaijaAgroChat":
        """
        Build the full pipeline.

        Args:
            docs_dir: Path to folder with agricultural PDFs/TXTs.
                      Defaults to Config.DOCS_DIR.
            force_rebuild_index: Rebuild FAISS even if a saved index exists.
            openai_api_key: OpenAI key for the safety layer (and optionally
                            the generator). Falls back to OPENAI_API_KEY env var.
            use_openai_generator: If True, use GPT-4o-mini as the generator
                                  instead of the local Aya model (useful for
                                  quick testing without a GPU).
        """
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
                    "No documents found.  Building an empty index. "
                    "Add PDFs to '%s' and rebuild.",
                    docs_dir,
                )
                # Create a minimal placeholder so the pipeline still runs
                from langchain_core.documents import Document as _Doc

                placeholder = _Doc(
                    page_content="No agricultural documents loaded yet.",
                    metadata={"source": "placeholder"},
                )
                chunks = [placeholder]
            vectorstore = build_or_load_vectorstore(chunks, force_rebuild=True)

        # ── Retriever ─────────────────────────────────────────────────────────
        retriever = build_retriever(vectorstore)

        # ── Generator ─────────────────────────────────────────────────────────
        if use_openai_generator:
            logger.info("Using OpenAI GPT-4o-mini as generator.")
            generator_llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0.1,
                api_key=openai_api_key,
            )
        else:
            logger.info(f"Loading local generator: {Config.GENERATOR_MODEL}")
            
            from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
            import torch
            from app.config.settings import dot_env_pth
   
            tokenizer = AutoTokenizer.from_pretrained(Config.GENERATOR_MODEL, 
                                                      token=dot_env_pth.HUGGINGFACE_API_KEY)
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
                max_new_tokens=512,
                do_sample=False,
            )
            from langchain_huggingface import HuggingFacePipeline

            generator_llm = HuggingFacePipeline(pipeline=pipe)

        # ── Safety LLM (OpenAI) ───────────────────────────────────────────────
        import os
        os.environ["OPENAI_API_KEY"] = dot_env_pth.OPENAI_API_KEY 
        safety_llm = ChatOpenAI(
            model=Config.OPENAI_SAFETY_MODEL,
            temperature=0,
            api_key=openai_api_key,
        )

        logger.info("NaijaAgroChat pipeline ready.")
        return cls(retriever, generator_llm, safety_llm)
