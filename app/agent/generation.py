from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# generation prompt templates used by NaijaAgroChat
GENERATION_SYSTEM = """You are NaijaAgroChat, a trusted agricultural assistant \
for Nigerian smallholder farmers. You MUST answer using ONLY the context \
provided below. You support queries in English, Hausa, Yoruba, Igbo, and \
Nigerian Pidgin — always respond in the SAME language the user used.

Rules:
- Base every answer strictly on the retrieved context.
- If the context does not contain enough information, say:
  "I don't have that information. Please consult your local extension officer."
- Never invent crop recommendations, pesticide dosages, or chemical names.
- Be concise and practical — farmers need actionable advice.
"""

GENERATION_HUMAN = """Context:
{context}

Question: {question}

Answer:"""


generation_prompt = ChatPromptTemplate.from_messages([
    ("system", GENERATION_SYSTEM),
    ("human", GENERATION_HUMAN),
])


def format_docs(docs: list[Document]) -> str:
    """Concatenate retrieved passages into a single context string."""
    return "\n\n---\n\n".join(
        f"[Source: {doc.metadata.get('source', 'unknown')}]\n{doc.page_content}"
        for doc in docs
    )
