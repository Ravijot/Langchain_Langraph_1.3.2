from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model("gpt-5.4")

class Emotion(BaseModel):
    emotion : str = Field(description="Emotion that is present in the string", default="Neutral")
    confidence_score : float = Field(description="How much confidence you have that this emotion is correct and range of this confidence score is between 0 to 10", default=0.0)

agent = create_agent(model=model, tools=[], response_format=Emotion, name="Agent-Emotion", system_prompt="You are a helpful assistant")

response = agent.invoke({"messages":[{"role":"user","content":"I am so exited to see cricket match today"}]})
print(response["structured_response"])

for message in response["messages"]:
    print(message.content)

