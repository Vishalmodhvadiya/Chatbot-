from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base

# ==========================================
# CHAT HISTORY DATABASE
# ==========================================

CHATBOT_DB_URL = (
    "postgresql://postgres:postgres@localhost:5435/chatbot_db"
)

chat_engine = create_engine(
    CHATBOT_DB_URL,
    pool_pre_ping=True
)

ChatSessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=chat_engine
)

# ==========================================
# COMPANY DATABASE
# ==========================================

COMPANY_DB_URL = (
    "postgresql://postgres:postgres@localhost:5435/company_db"
)

company_engine = create_engine(
    COMPANY_DB_URL,
    pool_pre_ping=True
)

CompanySessionLocal = sessionmaker(
    autocommit=False,
    autoflush=False,
    bind=company_engine
)

# ==========================================
# BASE MODEL
# ==========================================

Base = declarative_base()


# ==========================================
# DEPENDENCIES
# ==========================================

def get_chat_db():
    db = ChatSessionLocal()
    try:
        yield db
    finally:
        db.close()


def get_company_db():
    db = CompanySessionLocal()
    try:
        yield db
    finally:
        db.close()


SessionLocal = ChatSessionLocal