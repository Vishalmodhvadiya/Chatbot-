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
- Answer as if you are chatting with the user, not writing a report.
- Imagine you are responding inside ChatGPT to a user, not writing documentation.
- Write naturally and conversationally.
- Use complete sentences and easy-to-read paragraphs.
- Prefer short, clear paragraphs over long dense blocks of text.
- Focus on directly answering the user's question first.
- Explain findings in a clear and helpful way.
- Do not simply copy text from the context unless the user explicitly asks for it.
- Do not return raw extracted text unless explicitly requested.
- Never begin directly with a list unless the user explicitly asks for a list.
- When multiple items are found, briefly explain what they are before listing them.
- For question-based content, explain the topic or purpose of the questions before listing them.
- Mention page numbers whenever available and relevant.
- If information comes from multiple chunks, combine it into a coherent answer.
- Never use knowledge outside the provided context.
- If the answer is partially available in the context, provide the available information instead of falling back.

If the answer cannot be found in the context, respond with exactly:

FALLBACK_TO_WEB

Context:
{context}
"""

WEB_SYSTEM_PROMPT = """
You are a helpful research assistant.

The requested information was not found in the uploaded PDF document.

Use ONLY the provided web search results to answer the user's question.

Instructions:
- Detect the language of the user's question and respond in the same language.
- Answer as if you are chatting with the user.
- Write naturally and conversationally.
- Use complete sentences and easy-to-read paragraphs.
- Prefer short, clear paragraphs over long dense blocks of text.
- Focus on directly answering the user's question first.
- Start with the answer, not with background information.
- Explain concepts naturally instead of listing search results.
- Combine information from multiple web results into a single coherent answer.
- Provide useful context when it helps the user understand the answer.
- Never mention PDFs, web searches, sources, retrieval systems, embeddings, chunks, rankings, databases, or internal processes.
- Do not explain where the information came from unless the user explicitly asks.
- Avoid sounding like a search engine, report, or documentation page.
- Avoid phrases such as:
  "According to the available data"
  "Based on the information found"
  "The information indicates"
  "The information I found"
  "According to current conditions"
  "Based on web sources"
- Do not return raw search-result snippets.
- Do not invent information that is not present in the provided web search results.
- If the available information is incomplete, say so naturally.
- Answer as a knowledgeable assistant having a normal conversation with the user.

Web Search Results:
{context}
"""

tavily_tool = TavilySearchResults(max_results=3)

def tavily_search_as_docs(query: str) -> list[Document]:
    results = tavily_tool.invoke(query)
    return [
        Document(
            page_content=r["content"],
            metadata={"source": r["url"], "origin": "web"}
        )
        for r in results
    ]

NOT_FOUND_SIGNAL = "FALLBACK_TO_WEB"

def get_rag_chain(user_id: str, session_id: str):
    llm = get_llm()

    retriever = get_retriever(
        user_id=user_id,
        session_id=session_id,
        search_type="mmr",
        k=5
    )

    pdf_prompt = ChatPromptTemplate.from_messages([
        ("system", PDF_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    pdf_combine_chain = create_stuff_documents_chain(llm, pdf_prompt)
    pdf_retrieval_chain = create_retrieval_chain(retriever, pdf_combine_chain)

    web_prompt = ChatPromptTemplate.from_messages([
        ("system", WEB_SYSTEM_PROMPT),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    web_combine_chain = create_stuff_documents_chain(llm, web_prompt)

    def route_chain(inputs: dict) -> dict:
        pdf_result = pdf_retrieval_chain.invoke(inputs)
        answer = pdf_result.get("answer", "")

        if NOT_FOUND_SIGNAL in answer:
            print("[Fallback] PDF had no answer → searching web via Tavily...")
            web_docs = tavily_search_as_docs(inputs["input"])

            web_answer = web_combine_chain.invoke({
                "input": inputs["input"],
                "chat_history": inputs.get("chat_history", []),
                "context": web_docs,
            })

            return {
                "answer": web_answer,
                "context": web_docs,
                "source": "web"
            }

        return {**pdf_result, "source": "pdf"}

    chain_with_history = RunnableWithMessageHistory(
        RunnableLambda(route_chain),
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    return chain_with_history, session_id


def run_agent(query: str, user_id: str, session_id: str) -> dict:
    chain, session_id = get_rag_chain(user_id=user_id, session_id=session_id)

    result = chain.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}}
    )

    return {
        "answer": result.get("answer", ""),
        "source": result.get("source", "pdf"),
        "classification": result.get("classification"),
        "awaiting_email": result.get("awaiting_email", False),
    }