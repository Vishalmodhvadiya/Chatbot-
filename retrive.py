from langchain_core.vectorstores import get_vectorstore

def get_retriever(search_type="mmr", k=5):
    vectorstore = get_vectorstore()
    if search_type == "mmr":
        return vectorstore.as_retriever(
            search_type="mmr",
            search_kwargs={"k" : k, "fetch_k": 15}
        )

