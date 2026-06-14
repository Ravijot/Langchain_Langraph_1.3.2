from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain.tools import tool
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

model = init_chat_model("gpt-5")


@tool
def send_email(to: str, subject: str) -> str:
    """Send an email."""
    print(f"\nEMAIL SENT TO {to}")
    print(f"SUBJECT: {subject}")

    return "Email sent successfully"


agent = create_agent(
    model=model,
    tools=[send_email],
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "send_email": True
            }
        )
    ],
    checkpointer=InMemorySaver(),
)

config = {
    "configurable": {
        "thread_id": "demo-thread"
    }
}

result = agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": "Send an email to ceo@company.com saying Quarterly report ready"
            }
        ]
    },
    config=config,
    version="v2",
)
print(result)

inp = input("You want send mail or not. (Y/N)")

if inp == "Y":
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "approve"
                    }
                ]
            }
        ),
        config=config,
        version="v2",
    )
    print("###################################")
    for i in result.value["messages"]:
        print(f"{type(i)} : ",i.content)
else:
    result = agent.invoke(
        Command(
            resume={
                "decisions": [
                    {
                        "type": "reject",
                        "message": "User rejected this action. Do not retry this tool call.",
                    }
                ]
            }
        ),
        config=config,
        version="v2",
    )

    print("###################################")
    for i in result.value["messages"]:
        print(f"{type(i)} : ",i.content)