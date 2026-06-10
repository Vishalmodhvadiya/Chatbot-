import os
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import uuid

def load_and_chunk_pdf(file_bytes: bytes, user_id: str) -> list:

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name

    try:
      
        loader = PyPDFLoader(tmp_path)
        pages = loader.load()
      
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=2000,
            chunk_overlap=400,
            separators=["\n\n", "\n", ".", " "]
        )

        chunks = splitter.split_documents(pages)


        for chunk in chunks:
            chunk.metadata.update({
                "user_id": user_id
            })

       
        return chunks

    finally:
       
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)