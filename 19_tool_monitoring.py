from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from collections.abc import Callable
from langchain.agents.middleware import wrap_tool_call
from langchain.messages import ToolMessage
from langchain.tools.tool_node import ToolCallRequest
from langgraph.types import Command
from langchain.tools import tool

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

@wrap_tool_call
def monitor_tool(
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    print(f"Executing tool: {request.tool_call['name']}")
    print(f"Arguments: {request.tool_call['args']}")
    try:
        result = handler(request)
        print("Tool completed successfully")
        return result
    except Exception as e:
        print(f"Tool failed: {e}")
        raise

# Example tools
@tool
def population_info_of_city(city : str) -> int:
    """Get Population of particular city """
    return f"Population of {city} is 1 crore"

@tool
def weather(city: str) -> str:
    """Get weather information for a city."""
    print("Weather tool invoked")
    return f"The weather in {city} is sunny."

agent = create_agent(
    model=model,
    tools=[population_info_of_city, weather],
    middleware=[monitor_tool],
    system_prompt="You are a helpful assistant.",
)

response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Give population info of Delhi",
            }
        ]
    }
)


for message in response["messages"]:
    print(f"Message Type {type(message)} : ",message.content)