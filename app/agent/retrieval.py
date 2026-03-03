from .config import Config, logger

def build_retriever(vectorstore):
    """
    Two-stage retrieval:
      Stage 1 — FAISS approximate nearest-neighbour (k=10)
      Stage 2 — Cross-encoder re-ranker → top-5 passages
    Paper used XLM-RoBERTa-base cross-encoder fine-tuned for multilingual queries.
    """
    from langchain_community.cross_encoders import HuggingFaceCrossEncoder
    from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
    from langchain_classic.retrievers.document_compressors.cross_encoder_rerank import CrossEncoderReranker

    base_retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": Config.RETRIEVAL_K},
    )

    cross_encoder = HuggingFaceCrossEncoder(model_name=Config.RERANKER_MODEL)
    reranker = CrossEncoderReranker(model=cross_encoder, top_n=Config.RERANK_TOP_N)

    retriever = ContextualCompressionRetriever(
        base_compressor=reranker,
        base_retriever=base_retriever,
    )

    logger.info("Retriever with cross-encoder re-ranker ready.")
    return retriever
