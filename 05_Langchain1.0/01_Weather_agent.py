import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool

load_dotenv()

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

@tool('get_weather', description='Returns weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=metric&include=current&key={WEATHER_API_KEY}&contentType=json")
    return response

agent = create_agent(
    
    model=ChatGroq(model='openai/gpt-oss-20b', ),
    tools=[get_weather],
    system_prompt="""
    You are a helpful and funny weather assistant which always cracks jokes and 
    has a humourous response while also being remaining helpful."""   
)

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like in Indore today?'}
    ]
})

# print(response)
print(response['messages'][-1].content)