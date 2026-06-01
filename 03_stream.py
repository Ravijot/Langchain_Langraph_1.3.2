#By displaying output progressively, streaming significantly improves user experience, particularly for longer responses.
#stream() returns an iterator that yields output chunks as they are produced
#stream() returns multiple AIMessageChunk objects, each containing a portion of the output text. 
# Importantly, each chunk in a stream is designed to be gathered into a full message via summation

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4")

for chunk in model.stream("Write a small essay on AI"):
    print(chunk.text, end="", flush=True)


