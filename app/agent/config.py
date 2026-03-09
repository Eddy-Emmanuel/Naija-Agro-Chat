import logging

# configure root logger for the package
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
logger = logging.getLogger(__name__)


class Config:
    # Paths
    DOCS_DIR: str = "docs/"      # put FMARD/IITA/FAO PDFs here
    FAISS_INDEX_DIR: str = "faiss_index"

    # Chunking
    CHUNK_SIZE: int = 512
    CHUNK_OVERLAP: int = 64

    # Retrieval
    RETRIEVAL_K: int = 10          # candidates fetched before re-ranking
    RERANK_TOP_N: int = 5          # passages passed to the generator

    # Embedding model (paper: intfloat/multilingual-e5-large)
    EMBEDDING_MODEL: str = "intfloat/multilingual-e5-large"

    # Re-ranker cross-encoder (paper: XLM-RoBERTa-base cross-encoder)
    RERANKER_MODEL: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"

    # Generator (paper: CohereForAI/aya-23-8B)
    # For quick local testing swap to: "HuggingFaceH4/zephyr-7b-beta"
    GENERATOR_MODEL: str = "CohereForAI/aya-23-8B"

    MAX_NEW_TOKENS: int = 512       # tokens produced by the model
    MAX_TOTAL_TOKENS: int = 1024    # includes the prompt length

    # Safety layer
    OPENAI_SAFETY_MODEL: str = "gpt-4o-mini"      # cheap & fast for safety checks
    SAFETY_KEYWORDS: list[str] = [
        "pesticide", "herbicide", "fungicide", "insecticide",
        "dosage", "dose", "chemical", "spray", "litre", "liter",
        "ml", "gram", "kg", "application rate",
    ]

    # Web search fallback (requires SERPAPI_API_KEY in .env)
    USE_WEB_SEARCH: bool = True
    WEB_SEARCH_RESULTS: int = 5
