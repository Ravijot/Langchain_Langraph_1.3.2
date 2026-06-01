#Batching a collection of independent requests to a model can significantly improve performance and reduce costs, as the processing can be done in parallel

from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
import time

#Load environment variables from .env file
load_dotenv()

#Initialize the chat model
print("####### BATCH ######")
model = init_chat_model("gpt-5.4")


responses = model.batch([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
])

for response in responses:
    print(response)


print("###### BATCH AS COMPLETED #######")
for response in model.batch_as_completed([
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]):
    print(response)


 #When using batch_as_completed(), results may arrive out of order. Each includes the input index for matching to reconstruct the original order as needed.
list_of_inputs = [
    "Why do parrots have colorful feathers?",
    "How do airplanes fly?",
    "What is quantum computing?"
]
responses = model.batch(
    list_of_inputs,
    config={"max_concurrency": 5}
)

for response in responses:
    print(response.content)
