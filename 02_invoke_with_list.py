#A list of messages can be provided to a chat model to represent conversation history. 
#Each message has a role that models use to indicate who sent the message in the conversation

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

conversation = [
    {"role": "system", "content": "You are a helpful assistant that translates English to French."},
    {"role": "user", "content": "Translate: I love programming."},
    {"role": "assistant", "content": "J'adore la programmation."},
    {"role": "user", "content": "Translate: I love building applications."}
]

# from langchain.messages import HumanMessage, AIMessage, SystemMessage

# conversation = [
#     SystemMessage("You are a helpful assistant that translates English to French."),
#     HumanMessage("Translate: I love programming."),
#     AIMessage("J'adore la programmation."),
#     HumanMessage("Translate: I love building applications.")
# ]

response = model.invoke(conversation)
print(response.content)
