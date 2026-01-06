# %%
import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_community.utilities import SQLDatabase
from langchain_community.agent_toolkits import SQLDatabaseToolkit
from langchain.agents.middleware import HumanInTheLoopMiddleware 
from langgraph.checkpoint.memory import InMemorySaver 

# %%
checkpoint = InMemorySaver()

# %%
load_dotenv()

os.environ["LANGSMITH_TRACING"] = 'true'
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY")

# %%
db = SQLDatabase.from_uri("sqlite:///Sample.db")

print(f"Dialect: {db.dialect}")
print(f"Available tables: {db.get_usable_table_names()}")
print(f'Sample output: {db.run("SELECT * FROM Artist LIMIT 5;")}')

# %%
model = init_chat_model(model='openai/gpt-oss-20b', temperature=0.3, model_provider='groq')

# %%
sql_toolkit = SQLDatabaseToolkit(db=db, llm=model)

query_tools = sql_toolkit.get_tools()

for tool in query_tools:
    print(f"{tool.name}: {tool.description}\n")

# %%
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

# %%
agent = create_agent(
    model,
    query_tools,
    system_prompt=system_prompt,
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={"sql_db_query": True},
            description_prefix="Tool execution pending approval"
        )
    ],
    checkpointer=checkpoint,
)

# %%
question = "Which genre on average has the longest tracks?"

# %%
config = {"configurable": {"thread_id": "2"}} 

# %%
for step in agent.stream(
    {"messages": [{"role": "user", "content": question}]},
    config,
    stream_mode="values",
):
    if "messages" in step:
        step["messages"][-1].pretty_print()
    elif "__interrupt__" in step: 
        print("INTERRUPTED:") 
        interrupt = step["__interrupt__"][0] 
        for request in interrupt.value["action_requests"]: 
            print(request["description"]) 
    elif "interrupt" in step:
        print("\n=== INTERRUPTED ===")
        interrupt = step["interrupt"]
        for req in interrupt["value"]["action_requests"]:
            print(req["description"])
    else:
        print("*"*20)
        print(step)
        print("*"*20)
        pass

# %%



