NO_ANSWER_PHRASES = [
    "don't know", "not found", "no information",
    "cannot find", "i don't know", "not available",
    "don't have access", "context does not", "context doesn't",
    "unable to", "not mentioned", "no mention",
    "doesn't mention", "document does not", "not in the",
    "not provided", "no relevant", "couldn't find",
    "i was unable", "not contain", "doesn't contain",
    "don't have", "i'm not sure", "cannot answer",
    "does not provide", "no context", "provided context",
    "based on the context", "not discussed", "not covered",
    "not specified", "no details", "cannot provide",
    "real-time", "i don't have real", "current information",
    "live data", "up-to-date", "access to current",
    "does not mention", "not relate", "not relevant",
    "beyond the scope", "out of scope",
    "no_database_answer", "no database answer", "no database results",
]

def is_empty_answer(answer: str) -> bool:
    if not answer or len(answer.strip()) < 10:
        return True
    answer_lower = answer.lower()
    if "no_database_answer" in answer_lower or "no database answer" in answer_lower:
        return True
    return any(phrase in answer_lower for phrase in NO_ANSWER_PHRASES)
