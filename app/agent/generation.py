from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document

# generation prompt templates used by NaijaAgroChat
GENERATION_SYSTEM = """You are NaijaAgroChat, a trusted agricultural assistant \
for Nigerian smallholder farmers. You MUST answer using ONLY the context \
provided below. You support queries in English, Hausa, Yoruba, Igbo, and \
Nigerian Pidgin — always respond in the SAME language the user used.

If the user asks in a language other than English (e.g., Yoruba, Hausa, Igbo, or Nigerian Pidgin), answer in that language even if the retrieved context is in English. Do NOT translate your response into English.

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

# Translation prompt: used to translate non-English user queries into English for retrieval.
# If the input text is already English, the model should return it unchanged.
TRANSLATE_TO_ENGLISH_SYSTEM = """You are a translation assistant.
When given a user query, return only the English version of the text.
If the input is already in English, return it unchanged.
Do not add any extra commentary or formatting.
"""

TRANSLATE_TO_ENGLISH_HUMAN = """Text:
{text}

English translation:"""

translation_prompt = ChatPromptTemplate.from_messages([
    ("system", TRANSLATE_TO_ENGLISH_SYSTEM),
    ("human", TRANSLATE_TO_ENGLISH_HUMAN),
])

TRANSLATE_TO_QUERY_LANG_SYSTEM = """You are a translation assistant.
Given a user's original query and an English message, translate the message
into the language of the query. If the query is already in English, return
the message unchanged. Return only the translated message, with no extra
text or formatting.
"""

TRANSLATE_TO_QUERY_LANG_HUMAN = """Query:
{query}

Message:
{message}

Translated message:"""

translate_to_query_lang_prompt = ChatPromptTemplate.from_messages([
    ("system", TRANSLATE_TO_QUERY_LANG_SYSTEM),
    ("human", TRANSLATE_TO_QUERY_LANG_HUMAN),
])


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
