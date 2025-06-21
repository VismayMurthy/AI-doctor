import os
import requests
import json

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# Uncomment the following files if you're not using pipenv as your virtual environment manager
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

def call_ollama(prompt: str, model_name: str = "llama2") -> str:
    """Direct call to Ollama API."""
    try:
        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": model_name,
                "prompt": prompt,
                "stream": False
            },
            timeout=30
        )
        response.raise_for_status()
        return response.json()["response"]
    except Exception as e:
        return f"Error calling Ollama: {str(e)}"

# Step 2: Connect LLM with FAISS and Create chain

CUSTOM_PROMPT_TEMPLATE = """
Use the pieces of information provided in the context to answer user's question.
If you dont know the answer, just say that you dont know, dont try to make up an answer. 
Dont provide anything out of the given context

Context: {context}
Question: {question}

Start the answer directly. No small talk please.
"""

# Load Database
DB_FAISS_PATH="vectorstore/db_faiss"
embedding_model=HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
db=FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)

# Manual RAG implementation
user_query=input("Write Query Here: ")

# 1. Retrieve relevant documents
docs = db.similarity_search(user_query, k=3)
context = "\n".join([doc.page_content for doc in docs])

# 2. Create prompt
prompt = CUSTOM_PROMPT_TEMPLATE.format(
    context=context,
    question=user_query
)

# 3. Generate response using Ollama
result = call_ollama(prompt)

print("RESULT: ", result)
print("SOURCE DOCUMENTS: ", docs)