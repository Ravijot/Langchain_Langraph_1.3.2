from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.tools import tool
from langchain.agents import create_agent
from langchain.messages import HumanMessage

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

@tool(return_direct=False) # If return_direct=True agent will return the output as it is without giving to model and generating answers
def get_temperature(city : str) -> str:
    """
    This tool will help to find temperature of the city

    Args :
        city : This tool take name of city and provide temperature of that city
    """
    print("Room Temperature Tool Executed")
    return f"Temperature of {city} is 30 C"

tools = [get_temperature]

#Create Agent
agent = create_agent(model=model, tools=tools, system_prompt="You are a helpful assistant. Be concise and accurte")

response = agent.invoke({"messages": [{"role": "user", "content": "What the temperature of delhi"}]})
print(response)
print("########################")
for message in response["messages"]:
    print(message)