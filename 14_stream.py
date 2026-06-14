from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

agent = create_agent(
    model="openai:gpt-5.4",
    tools=[]
)

for chunk in agent.stream(
    {"messages": [{"role": "user", "content": "What is the weather in SF?"}]},
    stream_mode="messages",
    version="v2",
):
    if chunk["type"] == "messages":
        token, metadata = chunk["data"]

        if token.content:
            print(token.content, end="", flush=True)
