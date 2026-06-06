from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

agent = create_agent(model=model,tools=[], system_prompt="You are a helpful assistant.")

stream = agent.stream_events({"messages":[{"role":"user","content":"Write a short essay on AI"}]}, version="v3")

# for message in stream.messages:
#     for delta in message.text:
#         print(delta, end="", flush=True)

for message in stream.messages:
    
    for delta in message.text:
        print(delta, end="", flush=True)

    full_message = message.output
    usage = full_message.usage_metadata
    if usage:
        print(usage)

final_output = stream.output

print()
# print("Final Output : ", final_output)