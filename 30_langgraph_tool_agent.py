from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langchain.agents import create_agent
from langgraph.graph import START, END, StateGraph
from langchain.messages import AIMessage, ToolMessage
from typing_extensions import TypedDict


class State(TypedDict):
    user_input: str
    output: str
    tool_calls: list
    tool_message : str
    agent_called: str


class Graph:

    def __init__(self):
        load_dotenv()
        self.llm = init_chat_model(model="gpt-5.4")
        self.get_temperature_details_tool = tool(self.get_temperature_details)
        self.convert_dollars_into_rupees_tool = tool(self.convert_dollars_into_rupees)

    def get_temperature_details(self, city: str):
        """
        Useful in getting details of temperature of a particular city.
        """
        return f"Temperature of {city} is 30 degree Celsius"

    def convert_dollars_into_rupees(self, amount: str):
        """
        Convert dollars into rupees.
        """
        dollars = int(amount)
        rupees = dollars * 95
        return str(rupees)

    def weather_agent(self, state: State):
        query = state["user_input"]
        agent = create_agent(
            model=self.llm,
            tools=[self.get_temperature_details_tool],
            system_prompt="""
            You answer only using the available tools.
            """
        )
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        tool_record, tool_message = self.get_tool_call_record(response, "weather_agent")
        return {
            "tool_calls": tool_record,
            "output": response["messages"][-1].content,
            "agent_called" : "weather-agent",
            "tool_message" : tool_message
        }

    def currency_converter_agent(self, state: State):
        query = state["user_input"]
        state["agent_called"]="currency-converter"
        agent = create_agent(
            model=self.llm,
            tools=[self.convert_dollars_into_rupees_tool],
            system_prompt="""
            You answer only using the available tools.
            """
        )
        response = agent.invoke({"messages": [{"role": "user", "content": query}]})
        tool_record, tool_message = self.get_tool_call_record(response, "currency_converter_agent")
        return {
            "tool_calls": tool_record,
            "output": response["messages"][-1].content,
            "agent_called" : "currency_converter",
            "tool_message" : tool_message
        }

    def get_tool_call_record(self, response, agent):
        tool_records = []
        tool_message = ""
        for message in response["messages"]:
            if isinstance(message, AIMessage):
                for call in message.tool_calls:
                    tool_records.append(
                        {
                            "name": call["name"],
                            "args": call["args"],
                            "agent" : agent
                        }
                    )
            if isinstance(message, ToolMessage):
                tool_message = message.content

        return tool_records, tool_message

    def routing(self, state: State):
        query = state["user_input"].lower()
        if "dollar" in query or "rupee" in query:
            return "currency"
        if "weather" in query or "temperature" in query:
            return "weather"
        return END

    def build_workflow(self):

        graph = StateGraph(State)
        graph.add_node("weather", self.weather_agent)
        graph.add_node("currency",self.currency_converter_agent,)
        graph.add_conditional_edges(
            START,
            self.routing,
            {
                "weather": "weather",
                "currency": "currency",
                END: END,
            },
        )

        graph.add_edge("weather", END)
        graph.add_edge("currency", END)

        return graph.compile()


# Build graph
app = Graph().build_workflow()

# Run
result = app.invoke(
    {
        "user_input": "Give weather of delhi",
        "output": "",
        "tool_calls": [],
        "agent_called": "",
    }
)

print(result)