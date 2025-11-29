import os
import streamlit as st
from langchain_ollama.embeddings import OllamaEmbeddings
from langchain_groq.chat_models import ChatGroq
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv

load_dotenv()


groq_api_key = os.environ['GROQ_API_KEY']

if 'vectors' not in st.session_state:
    st.session_state.embeddings = OllamaEmbeddings(model='embeddinggemma:latest')
    st.session_state.loader = WebBaseLoader("https://docs.langchain.com/oss/python/langchain/install")
    st.session_state.docs = st.session_state.loader.load()
    
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size = 1000, chunk_overlap = 50)
    st.session_state.final_docs = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_docs, st.session_state.embeddings)
    

st.title("Chat Bot Interface")

llm = ChatGroq(api_key=groq_api_key, model='openai/gpt-oss-20b')
prompt = ChatPromptTemplate.from_template(
    """
    Answer the following questions based on the provided context only. Provide the most accurate response
    based on question.
    <context>{context}</context>
    Question : {input}
    """
)

retriever = st.session_state.vectors.as_retriever()

input = st.text_input("Input your prompt here")

if input:
    docs = retriever.invoke(input)
    context = "\n\n".join([d.page_content for d in docs])

    final_prompt = prompt.format(context=context, input=input)
    answer = llm.invoke(final_prompt)
    st.write(answer.content)
        
    