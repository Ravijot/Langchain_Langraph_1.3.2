# TOOL
#Tools extend what agents can do—letting them fetch real-time data, execute code, query external databases, and take actions in the world.

from langchain.tools import tool
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain.messages import HumanMessage

load_dotenv()

@tool
def get_temperature(city : str) -> str:
    """
    This tool will help to find temperature of the city

    Args :
        city : This tool take name of city and provide temperature of that city
    """
    print("Room Temperature Tool Executed")
    return f"Temperature of {city} is 30 C"

print(get_temperature.name)
print(get_temperature.description)

#Custoom Name and description of tool

@tool("get_rain_prediction",description="This tool will help to get information about rain prediction")
def get_rain_prediction(query :str)->str:
    print("Get Rain Prediction Tool Executed")
    return "Chances of Rain is 70%"

print(get_rain_prediction.name)
print(get_rain_prediction.description)

#Adding tool to LLM
model = init_chat_model("gpt-5.4")
model_with_tools = model.bind_tools([get_temperature,get_rain_prediction])
response = model_with_tools.invoke("Give tempeature of New Delhi")
print(response)
for tool_call in response.tool_calls:
    # View tool calls made by the model
    print(f"Tool: {tool_call['name']}")
    print(f"Args: {tool_call['args']}")

# messages = [{"role": "user", "content": "Give tempeature of New Delhi"}]
messages = [HumanMessage(content="Give tempeature of New Delhi")]
messages.append(response)

# Execute tools and collect results
for tool_call in response.tool_calls:
    # Execute the tool with the generated arguments
    tool_result = get_temperature.invoke(tool_call)
    messages.append(tool_result)

print(messages)

#Pass the result to final model

final_response = model_with_tools.invoke(messages)
print(final_response)
print("########### Final Response ##########")
print(final_response.text)