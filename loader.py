import tempfile
from langchain_core.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os

def load_and_chunk_pdf(uploaded_file) -> list:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    loader = PyPDFLoader(tmp_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=100,
    )
    chunks = splitter.split_documents(pages)

    for chunk in chunks:
        chunk.metadata["source_file"] = uploaded_file.name

    os.unlink(tmp_path)  
    return chunks