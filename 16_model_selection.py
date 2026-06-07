from langchain.agents import create_agent
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain.agents.middleware import ModelRequest, ModelResponse, wrap_model_call
from collections.abc import Callable

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
complex_model = init_chat_model("gpt-5.4")
simple_model = init_chat_model("gpt-5.1")
model = init_chat_model("gpt-5mini")

@wrap_model_call
def dynamic_model_selection(request : ModelRequest, handler : Callable[[ModelRequest], ModelResponse]) -> ModelResponse:

    if len(request.messages)>1:
        model = complex_model
    else:
        model = simple_model

    return handler(request.override(model=model))

agent = create_agent(name="AgentX", model=model, tools=[], middleware=[dynamic_model_selection], system_prompt="You are a helful assistant")

response = agent.invoke({"messages":[{"role":"user", "content":"hello"}]})

print(response["messages"][-1].response_metadata["model_name"])

response = agent.invoke({"messages":[{"role":"user","content":"hello "}, {"role":"assistant", "content":"How may i help you?"}, {"role":"user", "content":"tell me a joke"}]})
#If message length is more than the two messages it will select gpt 5.4
print(response["messages"][-1].response_metadata["model_name"])
