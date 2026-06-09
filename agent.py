import re
import warnings
warnings.filterwarnings("ignore")

from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_groq import ChatGroq
from llm import get_llm
from rag_chain import get_rag_chain
import os
from dotenv import load_dotenv

load_dotenv()

# ── Session store ──────────────────────────────────────────────────────────────

session_store: dict = {}

def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]

# ── User store (email per user) ────────────────────────────────────────────────

user_store: dict = {}

def get_user_email(user_id: str) -> str | None:
    return user_store.get(user_id, {}).get("email")

def set_user_email(user_id: str, email: str) -> None:
    user_store.setdefault(user_id, {})["email"] = email

# ── Pending query store (stores original query while awaiting email) ───────────

pending_store: dict = {}
# Structure: { user_id: { "query": "...", "file_id": "...", "session_id": "..." } }

def set_pending_query(user_id: str, query: str, session_id: str, file_id: str) -> None:
    pending_store[user_id] = {
        "query": query,
        "session_id": session_id,
        "file_id": file_id,
    }

def get_pending_query(user_id: str) -> dict | None:
    return pending_store.get(user_id)

def clear_pending_query(user_id: str) -> None:
    pending_store.pop(user_id, None)

# ── List store (sensitive topics per file_id) ──────────────────────────────────

list_store: dict = {}

def set_file_list(file_id: str, raw_text: str) -> None:
    topics = [line.strip() for line in raw_text.splitlines() if line.strip()]
    list_store[file_id] = topics

def get_file_list(file_id: str) -> list[str]:
    return list_store.get(file_id, [])

# ── Email extractor ────────────────────────────────────────────────────────────

EMAIL_REGEX = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")

def extract_email(text: str) -> str | None:
    match = EMAIL_REGEX.search(text)
    return match.group(0) if match else None

# ── Groq LLM for classifier ────────────────────────────────────────────────────

def get_classifier_llm():
    return ChatGroq(
        model="llama-3.3-70b-versatile",
        api_key=os.getenv("GROQ_API_KEY"),
        temperature=0.0
    )

# ── LLM1: query classifier ─────────────────────────────────────────────────────

CLASSIFIER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a semantic query classifier.

You will be given a user query and a list of sensitive topics.

Your job is to decide if the query is semantically related to ANY topic in the list.
This includes direct matches, synonyms, related concepts, or paraphrased meanings.

Examples of semantic matches:
- query "pricing"  → topic "cost"       → SENSITIVE
- query "costing"  → topic "investment" → SENSITIVE
- query "doctor"   → topic "medical"    → SENSITIVE
- query "attorney" → topic "legal"      → SENSITIVE

Respond with ONLY one word:
- SENSITIVE     (if the query semantically matches or relates to any topic in the list)
- NOT_SENSITIVE (if the query has no semantic relation to any topic in the list)

If the list is empty, always respond with NOT_SENSITIVE.

Sensitive topics:
{topics}
"""),
    ("human", "{query}"),
])

def classify_query(query: str, topics: list[str]) -> str:
    if not topics:
        return "NOT_SENSITIVE"
    llm = get_classifier_llm()
    chain = CLASSIFIER_PROMPT | llm
    result = chain.invoke({
        "query": query,
        "topics": "\n".join(f"- {t}" for t in topics),
    })
    text = result.content.strip().upper()
    return "NOT_SENSITIVE" if "NOT_SENSITIVE" in text else "SENSITIVE"

# ── Static messages ────────────────────────────────────────────────────────────

EMAIL_REQUEST_MESSAGE = (
    "To continue, please provide your email address. "
    "This helps us personalise your experience and keep your information secure."
)

# ── RAG runner helper ──────────────────────────────────────────────────────────

def _run_rag(query: str, user_id: str, session_id: str, file_id: str, classification: str, email_present: bool) -> dict:
    rag_chain, _ = get_rag_chain(
        user_id=user_id,
        session_id=session_id,
        file_id=file_id,
    )
    result = rag_chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    return {
        "answer": result.get("answer", result),
        "source": result.get("source", "pdf"),
        "classification": classification,
        "email_present": email_present,
        "awaiting_email": False,
    }

# ── Main agent entry point ─────────────────────────────────────────────────────

def run_agent(
    query: str,
    user_id: str,
    session_id: str,
    file_id: str = None,
) -> dict:
    """
    Flow:
    1. Check if query contains email → save it → run pending query automatically
    2. Email present → Groq classifies query → old flow runs
    3. No email + SENSITIVE   → save original query → ask for email
    4. No email + NOT_SENSITIVE → old flow runs directly
    """
    topics = get_file_list(file_id) if file_id else []

    # Step 1: check if query contains an email
    found_email = extract_email(query)
    if found_email:
        set_user_email(user_id, found_email)

        # check if there is a real question alongside the email
        clean_query = EMAIL_REGEX.sub("", query).strip().strip(".,!?- ")

        # use clean query if exists, otherwise use pending query
        if not clean_query:
            pending = get_pending_query(user_id)
            if pending:
                clear_pending_query(user_id)
                return _run_rag(
                    query=pending["query"],
                    user_id=user_id,
                    session_id=pending["session_id"],
                    file_id=pending["file_id"],
                    classification="SENSITIVE",
                    email_present=True,
                )
            return {
                "answer": "Thank you! Your email has been saved. Please go ahead with your question.",
                "source": "agent",
                "classification": None,
                "email_present": True,
                "awaiting_email": False,
            }
        else:
            query = clean_query

    stored_email = get_user_email(user_id)

    if stored_email:
        # ── Path A: email known ────────────────────────────────────────────────
        classification = classify_query(query, topics)
        return _run_rag(query, user_id, session_id, file_id, classification, email_present=True)

    else:
        # ── Path B: no email ───────────────────────────────────────────────────
        classification = classify_query(query, topics)

        if classification == "SENSITIVE":
            # save original query so we can run it after email is provided
            set_pending_query(user_id, query, session_id, file_id)
            return {
                "answer": EMAIL_REQUEST_MESSAGE,
                "source": "agent",
                "classification": "SENSITIVE",
                "email_present": False,
                "awaiting_email": True,
            }
        else:
            return _run_rag(query, user_id, session_id, file_id, classification, email_present=False)