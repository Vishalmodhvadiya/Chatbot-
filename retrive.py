from langchain_chroma import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_retriever(user_id: str, session_id: str, search_type: str = "mmr", k: int = 11):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory="./vectorstore_db"
    )

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "filter": {"user_id": {"$eq": user_id}}  # always filter by user_id
        }
    )

    return retriever