from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain_core.runnables import Runnable
from langgraph.store.postgres import PostgresStore

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

#LONG TERM MEMORY
# Long-term memory lets your agent store and recall information across different conversations and sessions. 
# Unlike short-term memory, which is scoped to a single thread, long-term memory persists across threads and can be recalled at any time.
# Long-term memory is built on LangGraph stores, which save data as JSON documents organized by namespace and key.
# InMemoryStore saves data to an in-memory dictionary.

DB_URI = "postgresql://postgres:postgres@localhost:5432/postgres?sslmode=disable"

with PostgresStore.from_conn_string(DB_URI) as store:
    store.setup()
    agent: Runnable = create_agent(
        "claude-sonnet-4-6",
        tools=[],
        store=store,
    )


result = agent.invoke({"messages":["role":"user","content":"Hello"]})

print(result)
