from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import InMemorySaver
from langchain_core.runnables import RunnableConfig
from typing import Annotated
from typing_extensions import TypedDict
from operator import add

class State(TypedDict):
    foo: str
    bar: Annotated[list[str], add]

def node_a(state: State):
    return {"foo": "a", "bar": ["a"]}

def node_b(state: State):
    return {"foo": "b", "bar": ["b"]}


workflow = StateGraph(State)
workflow.add_node(node_a)
workflow.add_node(node_b)
workflow.add_edge(START, "node_a")
workflow.add_edge("node_a", "node_b")
workflow.add_edge("node_b", END)

checkpointer = InMemorySaver()
graph = workflow.compile(checkpointer=checkpointer)

config: RunnableConfig = {"configurable": {"thread_id": "1"}}
graph.invoke({"foo": "", "bar":[]}, config)


# # get the latest state snapshot
# print("======== STATE =========")
# state = graph.get_state(config)
# print(state)
# checkpoint_id = f"{state.parent_config["configurable"]['checkpoint_id']}"

# # print(checkpoint_id)

# # get a state snapshot for a specific checkpoint_id
# print("=========== CHECKPOINT ============")
# config = {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint_id}}
# print(graph.get_state(config))


print("=========== HISTORY ============")
checkpoints = []
history = list(graph.get_state_history(config))
for snapshot in history:
    print(snapshot)
    print("VALUES : ",snapshot.values)
    print()
    checkpoints.append(snapshot.config["configurable"]['checkpoint_id'])

print("CHECKPOINTER : ",checkpoints)

for checkpoint in checkpoints:
    print()
    print("======== CHECPOINT ========")
    config = {"configurable": {"thread_id": "1", "checkpoint_id": checkpoint}}
    print(graph.get_state(config))
    print()