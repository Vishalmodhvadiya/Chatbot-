import warnings
warnings.filterwarnings("ignore")

from database import SessionLocal
from models import ChatMessage
from chat_history import save_message
from rag_chain import get_rag_chain
from agent import run_agent
from anthropic import Anthropic
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
import uuid
from llm import get_llm


# ── 1. Load history from DB ───────────────────────────────────────────────────

def get_chat_history_from_db(user_id: str, session_id: str) -> list[dict]:
    db = SessionLocal()
    try:
        rows = (
            db.query(ChatMessage)
            .filter(
                ChatMessage.user_id == user_id,
                ChatMessage.session_id == session_id,
            )
            .order_by(ChatMessage.created_at.asc())  # ← changed from message_time
            .all()
        )
        return [{"role": row.role, "content": row.message} for row in rows]  # ← here
    finally:
        db.close()

# ── 2. Summarize old history via Claude ───────────────────────────────────────

def summarize_history(history: list[dict]) -> str:
    history_text = "\n".join(
        f"{'User' if m['role'] == 'user' else 'Assistant'}: {m['content']}"
        for m in history
    )
    llm = get_llm()
    response = llm.invoke(f"Summarize this conversation briefly:\n\n{history_text}")
    return response.content

# ── 3. Compress if > 7 conversations ─────────────────────────────────────────

def compress_history(history: list[dict]) -> list[dict]:
    """
    If > 7 conversations (1 conv = 1 user + 1 assistant),
    summarize old ones and keep last 2 conversations.
    """
    total_conversations = len(history) // 2

    if total_conversations > 7:
        last_2 = history[-4:]       # last 2 conversations (4 messages)
        old_history = history[:-4]

        print(f"[Compressing: {total_conversations} conversations → summary + last 2]")
        summary = summarize_history(old_history)
        print(f"[Summary]: {summary}")

        return [
            {"role": "user",      "content": f"Summary of earlier conversation:\n{summary}"},
            {"role": "assistant", "content": "Understood, I have context of our earlier conversation."},
        ] + last_2

    return history


# ── 4. Inject compressed history into LangChain session_store ────────────────

def inject_history_into_langchain(session_id: str, history: list[dict]):
    """
    rag_chain.py uses get_session_history(session_id) from its own session_store.
    We import and populate it here so LangChain sees the DB history.
    """
    from rag_chain import session_store

    chat_history = ChatMessageHistory()
    for msg in history:
        if msg["role"] == "user":
            chat_history.add_message(HumanMessage(content=msg["content"]))
        else:
            chat_history.add_message(AIMessage(content=msg["content"]))

    session_store[session_id] = chat_history


# ── 5. Main chat function ─────────────────────────────────────────────────────

def chat_with_history(user_id: str, session_id: str, question: str) -> dict:
    """
    1. Load history from DB
    2. Compress if > 7 conversations
    3. Inject into LangChain session_store
    4. Run agent (RAG + classification + email logic)
    5. Save new messages to DB
    """
    # Load & compress
    history = get_chat_history_from_db(user_id, session_id)
    compressed = compress_history(history)

    # Inject into LangChain so rag_chain uses DB history
    inject_history_into_langchain(session_id, compressed)

    # Save user message
    save_message(user_id=user_id, session_id=session_id, role="user", message=question)

    # Run your existing agent (handles RAG + classification + email)
    result = run_agent(
        query=question,
        user_id=user_id,
        session_id=session_id,
    )

    answer = result["answer"]

    # Save assistant reply
    save_message(user_id=user_id, session_id=session_id, role="assistant", message=answer)

    print(f"[History: {len(history)//2} conversations, sent {len(compressed)//2} to LLM]")
    return result


# ── 6. CLI runner ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    user_id = input("Enter user_id (or press Enter for 'test_user'): ").strip() or "test_user"
    session_id = input("Enter session_id (or press Enter for new): ").strip() or str(uuid.uuid4())
    print(f"\nSession: {session_id}\nType 'quit' to exit.\n")

    while True:
        question = input("You: ").strip()
        if question.lower() in ("quit", "exit"):
            print("Bye!")
            break
        if not question:
            continue

        result = chat_with_history(user_id, session_id, question)
        print(f"Assistant: {result['answer']}\n")

