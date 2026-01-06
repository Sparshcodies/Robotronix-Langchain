import streamlit as st
import requests

def get_groq_response(input_text):
    response = requests.post(
        "http://localhost:8080/essay/invoke",
        json = { 
            "input" : {
                "topic":input_text
            }
        }
    )
    # print(response.json())
    return response.json()["output"]["content"]
    
def get_ollama_response(input_text):
    response = requests.post(
        "http://localhost:8080/poem/invoke",
        json = { 
            "input" : {
                "topic": input_text
            }
        }
    )
    # print(response.json())
    return response.json()["output"]

st.title("API Interaction")

input_text1 = st.text_input("Topic for essay")
input_text2 = st.text_input("Topic for poem")

if input_text1:
    st.write(get_groq_response(input_text1))
    
if input_text2:
    st.write(get_ollama_response(input_text2))
    