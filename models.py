from sqlalchemy import Column, BigInteger, String, Text, TIMESTAMP
from sqlalchemy.sql import func
from sqlalchemy.orm import declarative_base

Base = declarative_base()

class ChatMessage(Base):
    __tablename__ = "chatbot_messages"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(String(255), nullable=False)
    session_id = Column(String(255), nullable=False)
    user_message = Column(Text)
    assistant_reply = Column(Text)
    message_time = Column(TIMESTAMP, server_default=func.now())