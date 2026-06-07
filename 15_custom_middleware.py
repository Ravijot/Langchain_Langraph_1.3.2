from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import before_model, after_model, AgentState
from langgraph.runtime import Runtime
from langchain.messages import AIMessage
from typing import Any

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

"""LangChain’s create_agent runs on LangGraph’s runtime under the hood.
LangGraph exposes a Runtime object with the following information:
Context: static information like user id, db connections, or other dependencies for an agent invocation
Store: a BaseStore instance used for long-term memory
Stream writer: an object used for streaming information via the "custom" stream mode
Execution info: identity and retry information for the current execution (thread ID, run ID, attempt number)
Server info: server-specific metadata when running on LangGraph Server (assistant ID, graph ID, authenticated user)"""

# Hook	When it runs
# before_agent	Before agent starts (once per invocation)
# before_model	Before each model call
# after_model	After each model response
# after_agent	After agent completes (once per invocation)

# Hook	When it runs
# wrap_model_call	Around each model call
# wrap_tool_call	Around each tool call

@before_model(can_jump_to=["end"])
def check_message_limit(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    if len(state["messages"]) >= 3:
        print("Conversation Limit Reached")
        return {
            "messages": [AIMessage("Conversation limit reached.")],
            "jump_to": "end"
        }
    return None

@after_model
def log_response(state: AgentState, runtime: Runtime) -> dict[str, Any] | None:
    print(f"Model returned: {state['messages'][-1].content}")
    return None

agent = create_agent(name="AgentX", model=model, tools=[], system_prompt="You are a helpful assistant.", middleware=[check_message_limit,log_response])

response = agent.invoke({"messages":[{"role":"user", "content":"hello"},{"role":"assistant", "content":"How I can help you"}, {"role":"user", "content":"write a essay on AI"}]})

print(response)
