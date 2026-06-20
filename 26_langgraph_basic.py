from langgraph.graph import StateGraph, MessagesState, START, END
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model

load_dotenv()

model = init_chat_model(model="gpt-5.2")

def call_the_llm(state : MessagesState):
    response = model.invoke(state["messages"])
    return response

graph = StateGraph(MessagesState)

graph.add_node(call_the_llm)
graph.add_edge(START,"call_the_llm")
graph.add_edge("call_the_llm",END)

app = graph.compile()

result = app.invoke({"messages":[{"role":"user", "content": "hi"}]})
print(result)

#=============== OPTIONAL ===================
#TO DRAW GRAPH

from langchain_core.runnables.graph import CurveStyle, MermaidDrawMethod, NodeStyles
from PIL import Image as PILImage
import io

# Get the raw PNG bytes from Mermaid rendering
png_bytes = app.get_graph().draw_mermaid_png()

# Save it to a file
with open("graph_output.png", "wb") as f:
    f.write(png_bytes)

# Optional: open it using default image viewer
img = PILImage.open("graph_output.png")
img.show()
