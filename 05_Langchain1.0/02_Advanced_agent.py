import os
import requests
from dotenv import load_dotenv
from dataclasses import dataclass
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langgraph.checkpoint.memory import InMemorySaver
from langchain.tools import tool, ToolRuntime

load_dotenv()

WEATHER_API_KEY = os.getenv('WEATHER_API_KEY')
os.environ['GROQ_API_KEY'] = os.getenv('GROQ_API_KEY')

@dataclass
class Context:
    user_id : str
    
@dataclass
class ResponseFormat:
    summary: str
    temprature_fahrenhite: float
    temprature_celcius: float
    humidity: float
    
    
@tool('locate_user', description='Look up for user\'s city based on the context')
def locate_user(runtime: ToolRuntime[Context]):
    match runtime.context.user_id:
        case 'Sparsh':
            return 'Indore'
        case 'Keshav':
            return 'Bhopal'
        case 'Mahak':
            return 'Bengaluru'
        case _:
            return 'Unknown'
 
@tool('get_weather', description='Returns weather information for a given city', return_direct=False)
def get_weather(city: str):
    response = requests.get(f"https://weather.visualcrossing.com/VisualCrossingWebServices/rest/services/timeline/{city}?unitGroup=metric&include=current&key={WEATHER_API_KEY}&contentType=json")
    return response.json()

model = init_chat_model(model='llama-3.3-70b-versatile', temperature=0.3, model_provider='groq')
checkpoint = InMemorySaver()

agent = create_agent(
    model=model,
    tools=[get_weather,locate_user],
    system_prompt="""
    You are a helpful and funny weather assistant which always cracks jokes and 
    has a humourous response while also being remaining helpful.""",   
    context_schema=Context,
    response_format= ResponseFormat,
    checkpointer=checkpoint
)

config = {'configurable': {'thread_id': 1}}

response = agent.invoke({
    'messages': [
        {'role': 'user', 'content': 'What is the weather like today?'}
    ]},
    config=config,
    context=Context(user_id='Sparsh'))

# print(response['messages'][-1].content)
print(response['structured_response'])