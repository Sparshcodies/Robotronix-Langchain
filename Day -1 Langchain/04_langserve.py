from fastapi import FastAPI
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_ollama import OllamaLLM
from langserve import add_routes
from dotenv import load_dotenv
import uvicorn 
import os

load_dotenv()

os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"

app = FastAPI(title="Routing service for multiple llms")

model1 = ChatGroq(model="openai/gpt-oss-20b")
model2 = OllamaLLM(model='gemma3:1b')

prompt1 = ChatPromptTemplate.from_template("Write an essay on {topic} in about 500 words.")
prompt2 = ChatPromptTemplate.from_template("Write an poem on {topic} on about 300 words.")

add_routes(app, prompt1|model1, path="/essay")

add_routes(app, prompt2|model2, path="/poem")

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8080)