from fastapi import FastAPI, UploadFile, HTTPException, File, Form
from typing import Optional
from loader import load_and_chunk_pdf
from vector_db import add_documents
from agent import run_agent, set_file_list
from chat_history import save_message
import uuid

app = FastAPI()


@app.post("/upload/{user_id}")
async def upload_pdf(
    user_id: str,
    file: UploadFile = File(...),
    sensitive_list: Optional[str] = Form(default=None)
):
    file_bytes = await file.read()

    if not file_bytes.startswith(b"%PDF"):
        raise HTTPException(status_code=400, detail="Not a valid PDF")

    chunks = load_and_chunk_pdf(
        file_bytes=file_bytes,
        user_id=user_id
    )

    add_documents(chunks)

    if sensitive_list:
        set_file_list(user_id, sensitive_list)

    return {
        "file_name": file.filename,
        "chunks_count": len(chunks),
        "sensitive_list_loaded": bool(sensitive_list),
    }

@app.post("/chat/{user_id}")
async def chat(
    user_id: str,
    question: str = Form(...),
    session_id: Optional[str] = Form(None)
):
    if not session_id or session_id.strip() == "" or session_id.strip() == "string":
        session_id = str(uuid.uuid4())

    # Save user message
    save_message(
        user_id=user_id,
        session_id=session_id,
        role="user",
        message=question
    )

    result = run_agent(
        query=question,
        user_id=user_id,
        session_id=session_id,
    )

    answer = result["answer"]

    # Save assistant response
    save_message(
        user_id=user_id,
        session_id=session_id,
        role="assistant",
        message=answer
    )

    return {
        "question": question,
        "answer": answer,
        "session_id": session_id,
        "source_type": result.get("source"),
        "classification": result.get("classification"),
        "awaiting_email": result.get("awaiting_email", False),
    }