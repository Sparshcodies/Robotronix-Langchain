import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit

load_dotenv()

os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

model = init_chat_model(model='llama-3.3-70b-versatile', temperature=0.3, model_provider='groq')

db = SQLDatabase.from_uri("sqlite:///Sample.db")

sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)
query_tools = sql_toolkit.get_tools()


system_prompt = """
                You are an agent designed to interact with a SQL database.
                Given an input question, create a syntactically correct {dialect} query to run,
                then look at the results of the query and return the answer. Unless the user
                specifies a specific number of examples they wish to obtain, always limit your
                query to at most {top_k} results.
                You can order the results by a relevant column to return the most interesting
                examples in the database. Never query for all the columns from a specific table,
                only ask for the relevant columns given the question.
                You MUST double check your query before executing it. If you get an error while
                executing a query, rewrite the query and try again.
                DO NOT make any DML statements (INSERT, UPDATE, DELETE, DROP etc.) to the
                database. To start you should ALWAYS look at the tables in the database to see 
                what you can query. Do NOT skip this step.
                Then you should query the schema of the most relevant tables.
                """.format(dialect=db.dialect,top_k=5,)

agent = create_agent(
    model,
    query_tools,
    system_prompt=system_prompt,
)

question = "Which genre on average has the longest tracks?"
    
response = agent.invoke(
    {"messages": [{"role": "user", "content": question}]},
    stream_mode="values"
)

print(response['messages'][-1].content)