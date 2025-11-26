import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama.llms import OllamaLLM

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

# Prompt template
prompt = ChatPromptTemplate(
    [
        ("system","You are a helpful assistant. Please response to the user query"),
        ("user", "Question:{question}")
    ]
)

# Streamlit framework
st.title("Day 01 - Query using Ollama")
input_text = st.text_input("Enter a query")


# Method - 1
# from langchain_ollama.chat_models import ChatOllama
# llm = ChatOllama(model="gemma3:1b")

# Method - 2
# from langchain_community.llms.ollama import Ollama
# llm = Ollama(model="gemma3:1b")


llm = OllamaLLM(model="gemma3:1b")

output_parser = StrOutputParser()

chain = prompt|llm|output_parser

if input_text:
    st.write(chain.invoke({'question':input_text}))
    
