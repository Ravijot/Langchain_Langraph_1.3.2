from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.utils.uuid import uuid7
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

# model = init_chat_model("gpt-5.4")

agent = create_agent(model="openai:gpt-5.4", tools=[], name="AgentX", checkpointer=InMemorySaver(), system_prompt="You are a helpful assistant")

config = {"configurable": {"thread_id": str(uuid7())}}

#All agents include a sequence of messages in their state; to invoke the agent, pass a new message along with a thread_id so the agent can persist and resume conversation history
#Persisting conversation history with thread_id requires the agent to be configured with a checkpointer.

result1 = agent.invoke(
    {"messages": [{"role": "user", "content": "Yesterday, the weather was very cold and foggy"}]},
    config=config,
)

result2 = agent.invoke(
    {"messages": [{"role": "user", "content": "Give Info about yesterday's weather"}]},
    config=config,
)

# print("##### Result 1 #####")
# for message in result1["messages"]:
#     print(message.content)

print("##### Result 2 #####")
for message in result2["messages"]:
    print(message.content)