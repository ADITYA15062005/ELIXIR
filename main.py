from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import torch
import os
os.environ["STREAMLIT_DISABLE_WATCHDOG_WARNINGS"] = "true"
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"
from dotenv import load_dotenv
from langchain_community.llms import Ollama
import streamlit as st

# Load environment variables
load_dotenv()
st.toast("✅ Environment variables loaded.")

# Set LangChain tracing and API key
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
st.toast("✅ LangChain configuration set.")

# Set device for embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
st.toast(f"🖥 Torch device set to: {device}")

# Initialize HuggingFace Embeddings
embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2",
    model_kwargs={"device": device}
)
st.toast("🔎 HuggingFace Embeddings initialized.")

# Load FAISS index
db = FAISS.load_local(
    "faiss_index_alt/faiss_wine",
    embeddings=embedder,
    allow_dangerous_deserialization=True
)
st.toast("📁 FAISS index loaded from local storage.")

# Define the RAG prompt template
template = """
You are a wine connoisseur. The user wants to know: {question}

Here are some available wines:
{context}

Based on these, suggest the best fitting wine matching description.
"""

prompt = PromptTemplate(
    input_variables=["question", "context"],
    template=template
)
st.toast("📋 PromptTemplate initialized.")

# Initialize the LLM
llm = Ollama(model="llama3.2")
st.toast("🤖 Ollama LLM initialized.")

# Create the RetrievalQA chain
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=db.as_retriever(search_kwargs={"k": 3}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)
st.toast("🔗 RetrievalQA chain initialized.")

# Streamlit UI
st.title('Elixir – Where Taste Meets Technology.')
input_text = st.text_input("Search the topic you want")

if input_text:
    response = rag_chain.run(input_text)
    st.write(response)
    st.toast("✅ Query processed by RAG chain.")