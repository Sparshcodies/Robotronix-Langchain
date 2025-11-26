import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt template

prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please response to the user query"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework

st.title("Day 01 - Query using Groq API call")
input_text = st.text_input("Enter a query")

llm = ChatGroq(model="openai/gpt-oss-20b")
output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
    