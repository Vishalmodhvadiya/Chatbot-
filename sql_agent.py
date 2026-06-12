from sqlalchemy import text
from database import CompanySessionLocal
from llm import get_llm

SCHEMA = """
Tables:

actor(actor_id, first_name, last_name)

film(
    film_id,
    title,
    description,
    release_year,
    rental_rate,
    length,
    rating
)

film_actor(actor_id, film_id)

category(category_id, name)

film_category(film_id, category_id)

customer(
    customer_id,
    first_name,
    last_name,
    email
)

rental(
    rental_id,
    rental_date,
    customer_id,
    inventory_id,
    return_date
)

payment(
    payment_id,
    customer_id,
    rental_id,
    amount,
    payment_date
)

staff(
    staff_id,
    first_name,
    last_name,
    email
)

store(
    store_id,
    manager_staff_id
)

Relationships:

film_actor.actor_id -> actor.actor_id
film_actor.film_id -> film.film_id

film_category.film_id -> film.film_id
film_category.category_id -> category.category_id

rental.customer_id -> customer.customer_id

payment.customer_id -> customer.customer_id
payment.rental_id -> rental.rental_id
"""

FORBIDDEN_KEYWORDS = [
    "INSERT",
    "UPDATE",
    "DELETE",
    "DROP",
    "ALTER",
    "TRUNCATE",
    "CREATE",
    "GRANT",
    "REVOKE"
]


def clean_sql(sql: str) -> str:
    sql = (
        sql.replace("```sql", "")
        .replace("```", "")
        .strip()
    )

    if sql.endswith(";") is False:
        sql += ";"

    return sql


def validate_sql(sql_query: str):

    if not sql_query:
        raise Exception("Empty SQL generated")

    sql_upper = sql_query.upper()

    if not sql_upper.startswith("SELECT"):
        raise Exception("Only SELECT statements are allowed")

    for keyword in FORBIDDEN_KEYWORDS:
        if keyword in sql_upper:
            raise Exception(
                f"Forbidden SQL keyword detected: {keyword}"
            )


def query_company_db(question: str):

    db = CompanySessionLocal()

    try:

        llm = get_llm()

        print("\n========== COMPANY DB ==========")

        db_name = db.execute(
            text("SELECT current_database();")
        ).fetchone()

        print("DATABASE:", db_name)

        tables = db.execute(text("""
            SELECT table_name
            FROM information_schema.tables
            WHERE table_schema='public'
            ORDER BY table_name
        """)).fetchall()

        print("TABLE COUNT:", len(tables))

        print("================================\n")

        prompt = f"""
You are a PostgreSQL expert.

Database Schema:

{SCHEMA}

Rules:
- Return ONLY SQL.
- Return ONLY SELECT statements.
- Never use markdown.
- Never explain.
- Never use ```sql.
- Use only tables from schema.

Question:
{question}
"""

        sql_query = llm.invoke(prompt).content.strip()

        sql_query = clean_sql(sql_query)

        print("\n==============================")
        print("QUESTION:")
        print(question)

        print("\nGENERATED SQL:")
        print(sql_query)
        print("==============================\n")

        validate_sql(sql_query)

        rows = db.execute(
            text(sql_query)
        ).fetchall()

        print("ROWS RETURNED:")
        print(rows)

        if not rows:
            return {
                "found": False
            }

        answer_prompt = f"""
User Question:
{question}

Database Result:
{rows}

Answer naturally.


Rules:
- Use the exact values from Database Result.
- Do not count rows returned.
- If result is [(200,)], answer that the count is 200.
- Never invent numbers.
- Do not mention SQL.
- Do not mention database.
- Answer directly.
"""

        answer = llm.invoke(answer_prompt).content

        return {
            "found": True,
            "answer": answer,
            "source": "database",
            "sql": sql_query
        }

    except Exception as e:

        print("\n[SQL AGENT ERROR]")
        print(str(e))

        return {
            "found": False,
            "error": str(e)
        }

    finally:
        db.close()