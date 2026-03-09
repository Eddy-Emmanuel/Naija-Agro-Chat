from pathlib import Path
from typing import List, Optional

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    PyPDFLoader,
    DirectoryLoader,
    TextLoader,
)

from .config import Config, logger


def load_documents(docs_dir: str) -> List[Document]:
    """Load PDFs and plain-text files from the corpus directory."""
    docs_path = Path(docs_dir)
    if not docs_path.exists():
        logger.warning(f"Docs directory '{docs_dir}' not found — using empty corpus.")
        return []

    loaders = []

    # PDF loader
    pdf_loader = DirectoryLoader(
        docs_dir,
        glob="*.pdf",
        loader_cls=PyPDFLoader,
        show_progress=True,
    )
    loaders.append(pdf_loader)

    documents: List[Document] = []
    for loader in loaders:
        try:
            documents.extend(loader.load())
        except Exception as e:
            logger.warning(f"Loader error: {e}")

    logger.info(f"Loaded {len(documents)} raw documents.")
    return documents


def chunk_documents(documents: List[Document]) -> List[Document]:
    """
    Semantic-ish chunking using RecursiveCharacterTextSplitter.
    Paper used ~512-token chunks; we approximate with character counts
    (1 token ≈ 4 chars for English, slightly less for African languages).
    """
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=Config.CHUNK_SIZE * 4,   # chars → ~512 tokens
        chunk_overlap=Config.CHUNK_OVERLAP * 4,
        separators=["\n\n", "\n", ".", " ", ""],
        length_function=len,
    )
    chunks = splitter.split_documents(documents)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks


def build_or_load_vectorstore(
    chunks: Optional[List[Document]] = None,
    force_rebuild: bool = False,
) -> "FAISS":
    """
    Build a FAISS index from chunks, or load an existing one from disk.
    Embedding model: intfloat/multilingual-e5-large (multilingual, 1024-dim).
    """
    from langchain_community.vectorstores import FAISS
    from langchain_huggingface import HuggingFaceEmbeddings

    embeddings = HuggingFaceEmbeddings(
        model_name=Config.EMBEDDING_MODEL,
        model_kwargs={"device": "cuda"},          # change to "cuda" if available
        encode_kwargs={"normalize_embeddings": True},
    )

    index_path = Path(Config.FAISS_INDEX_DIR)

    if index_path.exists() and not force_rebuild:
        logger.info("Loading existing FAISS index from disk …")
        vectorstore = FAISS.load_local(
            str(index_path),
            embeddings,
            allow_dangerous_deserialization=True,
        )
        logger.info("FAISS index loaded.")
    else:
        if not chunks:
            raise ValueError("chunks must be provided when building a new index.")
        logger.info("Building FAISS index …")
        vectorstore = FAISS.from_documents(chunks, embeddings)
        vectorstore.save_local(str(index_path))
        logger.info(f"FAISS index saved to '{Config.FAISS_INDEX_DIR}'.")

    return vectorstore
