from sqlalchemy import create_engine, text

engine = create_engine(
    "postgresql://postgres:postgres@localhost:5435/company_db"
)

with engine.connect() as conn:
    result = conn.execute(text("SELECT current_database();"))
    print(result.fetchone())