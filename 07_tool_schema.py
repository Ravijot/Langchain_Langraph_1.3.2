from langchain.tools import tool
from typing import Literal
from pydantic import BaseModel, Field

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

print("Name of tool : ",get_weather_details.name)
print("Description of tool : ",get_weather_details.description)
print("Arguements of the tool : ",get_weather_details.args)


print(get_weather_details.invoke({"location":"Delhi","units":"celcius"}))

# Invoking tool with wrong arguements
try:
    get_weather_details.invoke({"location":"Delhi","units":"someUnit"})
except Exception as e:
    print("Exception :",str(e))