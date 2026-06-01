#Install the required libraries
# pip install langchain-openai
# pip install python-dotenv
# pip install langchain==1.3.2
#create a .env folder in root and add key like this in env OPENAI_API_KEY="Your KEY"
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
model = init_chat_model("gpt-5.4", temperature=0.7, max_tokens=1000, max_retries=6)

#Pass a prompt to the model and get response
response = model.invoke("hello")

#Print the response
print(response)