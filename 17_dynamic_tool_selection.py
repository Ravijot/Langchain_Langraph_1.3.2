from dotenv import load_dotenv
from typing import Callable

from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain.tools import tool

# Load environment variables
load_dotenv()

# Initialize model
model = init_chat_model("gpt-5")

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

ALL_TOOLS = [population_info_of_city, weather]


def select_relevant_tools(state, runtime):
    """
    Simple example of dynamic tool selection.
    Replace this with your own logic.
    """
    messages = state.get("messages", [])

    if not messages:
        return []

    last_message = messages[-1].content.lower()

    if any(word in last_message for word in ["population"]):
        return [population_info_of_city]

    if any(word in last_message for word in ["weather", "temperature", "rain"]):
        return [weather]

    return ALL_TOOLS


@wrap_model_call
def select_tools(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:
    """Select relevant tools before each model call."""

    relevant_tools = select_relevant_tools(
        request.state,
        request.runtime,
    )

    return handler(request.override(tools=relevant_tools))


agent = create_agent(
    model=model,
    tools=ALL_TOOLS,
    middleware=[select_tools],
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