from models import ChatMessage
from database import SessionLocal

def save_message(user_id, session_id, role, message):
    db = SessionLocal()
    try:
        db.add(ChatMessage(
            user_id=user_id,
            session_id=session_id,
            role=role,
            message=message
        ))
        db.commit()
    except Exception as e:
        db.rollback()
        print("ERROR:", e)
    finally:
        db.close()