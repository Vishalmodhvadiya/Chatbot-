from langchain_chroma import Chroma
from embedding import get_embeddings

PERSIST_DIR = "vectorstore_db"
COLLECTION_NAME = "pdf_chunks" 

def get_vectorstore():
    return Chroma(
        collection_name=COLLECTION_NAME,  
        persist_directory=PERSIST_DIR,
        embedding_function=get_embeddings()
    )

def add_documents(chunks):
    vectorstore = get_vectorstore()
    vectorstore.add_documents(chunks)
    return vectorstore