from typing import Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

from .config import Config, logger

SAFETY_SYSTEM = """You are a safety verifier for agricultural advice given to \
Nigerian farmers. Your job is to check whether the ANSWER below contains \
specific chemical names, pesticide application rates, dosages, or other \
safety-critical recommendations that are NOT explicitly supported by the \
provided CONTEXT.

Respond with ONLY a JSON object (no markdown, no extra text):
{
  "safe": true | false,
  "reason": "<one-sentence explanation>"
}

safe=true  → the answer is grounded in the context and does not contain \
             unsupported chemical/dosage claims.
safe=false → the answer makes specific chemical or dosage claims that are \
             not found in the context.
"""

SAFETY_HUMAN = """CONTEXT:
{context}

ANSWER:
{answer}
"""

safety_prompt = ChatPromptTemplate.from_messages([
    ("system", SAFETY_SYSTEM),
    ("human", SAFETY_HUMAN),
])


def is_safety_critical(query: str) -> bool:
    """Quick heuristic: does the query mention safety-sensitive topics?"""
    q_lower = query.lower()
    return any(kw in q_lower for kw in Config.SAFETY_KEYWORDS)


def verify_safety(
    query: str,
    answer: str,
    context: str,
    safety_llm: ChatOpenAI,
) -> Tuple[bool, str]:
    """
    Use OpenAI to verify that the answer does not contain unsupported
    chemical/dosage recommendations.  Returns (is_safe, reason).
    """
    import json

    chain = safety_prompt | safety_llm | StrOutputParser()
    raw = chain.invoke({"context": context, "answer": answer})

    # Strip potential markdown fences
    raw = raw.strip().lstrip("```json").lstrip("```").rstrip("```").strip()

    try:
        result = json.loads(raw)
        return bool(result.get("safe", False)), result.get("reason", "")
    except json.JSONDecodeError:
        logger.warning(f"Safety check returned non-JSON: {raw}")
        # Conservative: treat as unsafe if we can't parse
        return False, "Could not parse safety check response."
