from langchain.tools import tool
from typing import Literal
from pydantic import BaseModel, Field
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-5.4")

class WeatherInput(BaseModel):
    """Input for Weather Queries"""
    location : str = Field(description="City Name")
    units : Literal["celcius","fahernheit"] = Field(
        default="celcius",
        description="Temperature unit preference"
    )

@tool(args_schema=WeatherInput)
def get_weather_details(location : str, units : str = "celcius"):
    """Get Current Weather Information"""

    return f"Current temperature {location} is 25 {units}"


agent = create_agent(name="Weather-Agent", model=model, system_prompt="You are a helpful assistant", tools=[get_weather_details])

response = agent.invoke({"messages":[{"role":"user","content":"What's the temperature in Delhi?"}]})

for message in response["messages"]:
    print(message.content)
    print(type(message))
