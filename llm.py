from langchain_groq import Groq
from dotenv import load_dotenv
import os

load_dotenv()

def get_llm(model_name="llama-2-8b-chat-hf"):
    return Groq(
        groq_api_key=os.getenv("GROQ_API_KEY"),
        model=model_name,
    )