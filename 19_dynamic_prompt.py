from collections.abc import Callable

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain.agents.middleware import (
    wrap_model_call,
    ModelRequest,
    ModelResponse,
)
from langchain.chat_models import init_chat_model
from langchain.messages import SystemMessage

load_dotenv()

model = init_chat_model("gpt-5.1")


@wrap_model_call
def add_customer_support_context(
    request: ModelRequest,
    handler: Callable[[ModelRequest], ModelResponse],
) -> ModelResponse:

    messages = request.state.get("messages", [])
    latest_message = messages[-1].content.lower()

    if any(
        word in latest_message
        for word in ["angry", "terrible", "worst", "refund", "cancel", "frustrated"]
    ):
        dynamic_instruction = """
Customer appears unhappy.

- Acknowledge their frustration.
- Apologize for the inconvenience.
- Prioritize solving the problem.
- Keep the tone professional and empathetic.
"""
    else:
        dynamic_instruction = """
Customer appears calm.

- Be friendly and concise.
- Answer the question directly.
"""

    print("\n========== DYNAMIC PROMPT ==========")
    print(dynamic_instruction)

    new_content = list(request.system_message.content_blocks)

    new_content.append(
        {
            "type": "text",
            "text": dynamic_instruction,
        }
    )

    new_system_message = SystemMessage(content=new_content)

    return handler(
        request.override(
            system_message=new_system_message
        )
    )


agent = create_agent(
    model=model,
    tools=[],
    middleware=[add_customer_support_context],
    system_prompt="""
You are a customer support agent for an e-commerce company.
""",
)

# Customer is upset
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "This is the worst experience ever. I want a refund."
            }
        ]
    }
)

print("\nAssistant:")
print(response["messages"][-1].content)


# Customer is normal
response = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "When will my order arrive?"
            }
        ]
    }
)


for message in response["messages"]:
    print(f"Message Type {type(message)} : ",message.content)