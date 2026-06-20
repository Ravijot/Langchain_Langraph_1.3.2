from langchain.messages import AnyMessage
from typing_extensions import TypedDict, Annotated
import operator
from langgraph.graph import StateGraph, START, END
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(
    "gpt-5",
    model_provider="openai"
)


class SupportState(TypedDict):
    messages: Annotated[list[AnyMessage], operator.add]
    llm_calls: int
    customer_id: str | None
    customer_name: str | None
    ticket_description: str | None
    issue_category: str | None


def find_customer_details(state: SupportState):

    customer_database = {
        "345678": {"customer_name": "Corey"},
        "789047": {"customer_name": "Karisa"}
    }

    if state["customer_id"] in customer_database:
        return {
            "customer_name":
            customer_database[state["customer_id"]]["customer_name"]
        }

    return {"customer_name": "No Data Found"}


def preprocess_the_query(state: SupportState):

    query = state["ticket_description"].strip().lower()

    return {
        "ticket_description": query
    }


def classify_the_ticket_query(state: SupportState):

    classify_prompt = PromptTemplate(
        template="""
You are a helpful assistant.
Classify the query into one category only.

Categories:
["Infrastructure",
"Laptop/Asset",
"Software Issue",
"Security Concerns"]

Question: {question}
""",
        input_variables=["question"]
    )

    chain = classify_prompt | model

    response = chain.invoke({
        "question": state["ticket_description"]
    })

    return {
        "issue_category": response.content.strip()
    }


graph = StateGraph(SupportState)

graph.add_node("preprocess", preprocess_the_query)
graph.add_node("customer", find_customer_details)
graph.add_node("classifier", classify_the_ticket_query)

graph.add_edge(START, "preprocess")
graph.add_edge("preprocess", "customer")
graph.add_edge("customer", "classifier")
graph.add_edge("classifier", END)

app = graph.compile()

result = app.invoke({
    "messages": [],
    "llm_calls": 0,
    "customer_id": "345678",
    "ticket_description": "I am not able to open my laptop"
})

print(result)


#=============== OPTIONAL ===================
#TO DRAW GRAPH

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = app.get_graph().draw_mermaid_png()

# Save it to a file
with open("27_langgraph.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("27_langgraph.png")
img.show()