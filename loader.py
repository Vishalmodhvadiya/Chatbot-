import tempfile
from langchain_core.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_files_and_chunk(uploaded_file):

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    chunk_size=1000
    chunk_overlap=50
    
    loader = PyPDFLoader(tmp_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    chunks = text_splitter.split_documents(documents)
   
    chunks = text_splitter.split_documents(pages_content)

   
    for chunk in chunks:
        chunk.metadata["source_file"] = uploaded_file.name

    os.unlink(tmp_path)  
    return chunks






