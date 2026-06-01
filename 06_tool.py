# TOOL
#Tools extend what agents can do—letting them fetch real-time data, execute code, query external databases, and take actions in the world.

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")