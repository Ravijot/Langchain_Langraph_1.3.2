from typing_extensions import Literal, TypedDict
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from langchain.messages import HumanMessage, SystemMessage
from langchain.chat_models import init_chat_model

from langgraph.graph import StateGraph, START, END

load_dotenv()

# Initialize model
model = init_chat_model(model="gpt-5")

# Structured output schema
class Route(BaseModel):
    step: Literal[
        "fetch_account_balance",
        "block_atm",
        "get_user_info",
        "other",
    ] = Field(
        description="The next step in the routing process"
    )

# Create structured output model
router_llm = model.with_structured_output(Route)


# Graph state
class State(TypedDict):
    input: str
    decision: str
    output: str


# ----------------------
# Tool Nodes
# ----------------------
def fetch_account_balance(state: State):
    """Helpful for fetching account balance"""
    return {
        "output": "You have 20,000 Rupees in your account."
    }


def block_atm(state: State):
    """Helpful in blocking ATM card"""
    return {
        "output": "Your ATM card has been blocked successfully."
    }


def get_user_info(state: State):
    """Helpful in getting user information"""
    return {
        "output": (
            "Username: John\n"
            "Account Number: 1245222\n"
            "Balance: 20,000"
        )
    }


def other(state: State):
    """Fallback node"""
    return {
        "output": "Sorry, I cannot help with that request."
    }


# ----------------------
# Router Node
# ----------------------
def llm_call_router(state: State):
    decision = router_llm.invoke(
        [
            SystemMessage(
                content=(
                    "Route the user's request to one of these options:\n"
                    "- fetch_account_balance\n"
                    "- block_atm\n"
                    "- get_user_info\n"
                    "- other\n\n"
                    "Return only the appropriate route."
                )
            ),
            HumanMessage(content=state["input"]),
        ]
    )

    return {
        "decision": decision.step
    }


# ----------------------
# Conditional Routing
# ----------------------
def route_decision(state: State):
    print(state["decision"])
    return state["decision"]


# ----------------------
# Build Graph
# ----------------------
route_builder = StateGraph(State)

# Add nodes
route_builder.add_node("llm_call_router", llm_call_router)
route_builder.add_node("fetch_account_balance",fetch_account_balance)
route_builder.add_node("block_atm",block_atm)
route_builder.add_node("get_user_info",get_user_info)
route_builder.add_node("other",other)

# Start → Router
route_builder.add_edge(START, "llm_call_router")

# Router → Next node
route_builder.add_conditional_edges(
    "llm_call_router",
    route_decision,
    { # Name returned by route_decision : Name of next node to visit
        "fetch_account_balance": "fetch_account_balance",
        "block_atm": "block_atm",
        "get_user_info": "get_user_info",
        "other": "other",
    },
)

# End edges
route_builder.add_edge("fetch_account_balance", END)
route_builder.add_edge("block_atm", END)
route_builder.add_edge("get_user_info", END)
route_builder.add_edge("other", END)

# Compile graph
router_workflow = route_builder.compile()


# ----------------------
# Invoke
# ----------------------
result = router_workflow.invoke(
    {
        "input": "I want to know my balance"
    }
)

print(result["output"])

#=============== OPTIONAL ===================
#TO DRAW GRAPH

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = router_workflow.get_graph().draw_mermaid_png()

# Save it to a file
with open("28_langgraph_routing.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("28_langgraph_routing.png")
img.show()