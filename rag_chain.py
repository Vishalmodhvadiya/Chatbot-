import warnings
warnings.filterwarnings("ignore")

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.runnables import RunnableLambda
from langchain_core.documents import Document

from llm import get_llm
from retrive import get_retriever
from sql_agent import query_company_db


session_store = {}
sensitive_store = {}


def get_session_history(session_id: str) -> ChatMessageHistory:
    if session_id not in session_store:
        session_store[session_id] = ChatMessageHistory()
    return session_store[session_id]


def set_file_list(user_id: str, sensitive_list: str):
    sensitive_store[user_id] = sensitive_list


PDF_SYSTEM_PROMPT = """
You are a helpful research assistant analyzing a PDF document.

Instructions:
- Answer ONLY using the provided context.
- Detect the language of the user's question and respond in the same language.
- Answer naturally and conversationally.
- Focus on directly answering the user's question.
- Mention page numbers whenever available.
- Never use knowledge outside the provided context.

If the answer cannot be found in the context, respond with exactly:


NOT_FOUND

Context:
{context}
"""


WEB_SYSTEM_PROMPT = """
You are a helpful research assistant.

The requested information was not found in the uploaded PDF document.

Use ONLY the provided web search results to answer the user's question.

Instructions:
- Detect the language of the user's question and respond in the same language.
- Answer naturally and conversationally.
- Focus on directly answering the user's question.
- Never mention retrieval systems, PDFs, embeddings, databases, or search tools.

If the answer is not present in the web search results,
respond with exactly:

NOT_FOUND

Web Search Results:
{context}
"""


tavily_tool = TavilySearchResults(max_results=3)


def tavily_search_as_docs(query: str) -> list[Document]:
    results = tavily_tool.invoke(query)

    return [
        Document(
            page_content=r["content"],
            metadata={
                "source": r["url"],
                "origin": "web"
            }
        )
        for r in results
    ]


NOT_FOUND_SIGNAL = "NOT_FOUND"


def get_rag_chain(user_id: str, session_id: str):

    llm = get_llm()

    retriever = get_retriever(
        user_id=user_id,
        session_id=session_id,
        search_type="mmr",
        k=5
    )

    # PDF CHAIN

    pdf_prompt = ChatPromptTemplate.from_messages([
        ("system", PDF_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    pdf_combine_chain = create_stuff_documents_chain(
        llm,
        pdf_prompt
    )

    pdf_retrieval_chain = create_retrieval_chain(
        retriever,
        pdf_combine_chain
    )

    # WEB CHAIN

    web_prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}")
    ])

    web_combine_chain = create_stuff_documents_chain(
        llm,
        web_prompt
    )

    # ROUTER

    def route_chain(inputs: dict) -> dict:

        question = inputs["input"]

    # =====================================
    # STEP 1 : PDF
    # =====================================

        pdf_result = pdf_retrieval_chain.invoke(inputs)

        print("\n[PDF RESULT]")
        print(pdf_result.get("answer", ""))
        print()

        answer = str(pdf_result.get("answer", "")).strip()

        if answer.strip() != NOT_FOUND_SIGNAL:
            return {
                **pdf_result,
                "source": "pdf"
            }
 
    # =====================================
    # STEP 2 : DATABASE
    # =====================================

        try:
           db_result = query_company_db(question)

           print("\n[DATABASE RESULT]")
           print(db_result)
           print()

           if db_result and db_result.get("found", False):
               return {
                 "answer": db_result["answer"],
                 "source": "database",
                 "context": []
              }

        except Exception as e:
            print(f"[DATABASE ERROR] {e}")
 
    # =====================================
    # STEP 3 : WEB
    # =====================================

        print("[Fallback] PDF & Database had no answer → searching web")

        web_docs = tavily_search_as_docs(question)

        web_answer = web_combine_chain.invoke({
          "input": question,
          "chat_history": inputs.get("chat_history", []),
          "context": web_docs,
        })

        web_text = str(web_answer).strip()
        if web_text.strip() != NOT_FOUND_SIGNAL:
            return {
                "answer": web_answer,
                "context": web_docs,
                "source": "web"
            }
 
    # =====================================
    # STEP 4 : NOTHING FOUND
    # =====================================

        return {
            "answer": "I couldn't find information.",
            "source": "none",
            "context": []
        }

  

    chain_with_history = RunnableWithMessageHistory(
        RunnableLambda(route_chain),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain_with_history, session_id


def run_agent(
    query: str,
    user_id: str,
    session_id: str
) -> dict:

    chain, session_id = get_rag_chain(
        user_id=user_id,
        session_id=session_id
    )

    result = chain.invoke(
        {"input": query},
        config={
            "configurable": {
                "session_id": session_id
            }
        }
    )

    return {
        "answer": result.get("answer", ""),
        "source": result.get("source", "pdf"),
        "classification": result.get("classification"),
        "awaiting_email": result.get("awaiting_email", False),
    }