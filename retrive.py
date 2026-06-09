from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

def get_retriever(user_id: str, session_id: str, file_id: str = None, search_type: str = "mmr", k: int = 11):
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    
    vectorstore = Chroma(
        collection_name="pdf_chunks",
        embedding_function=embeddings,
        persist_directory="./vectorstore_db"
    )

    if file_id:
        where_filter = {"file_id": {"$eq": file_id}}
    else:
        where_filter = {"user_id": {"$eq": user_id}}

    retriever = vectorstore.as_retriever(
        search_type=search_type,
        search_kwargs={
            "k": k,
            "filter": where_filter
        }
    )

    return retriever