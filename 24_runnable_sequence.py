import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import  RunnableLambda

load_dotenv()
    
model = init_chat_model("gpt-4o-mini")
parser = StrOutputParser()

sentiment_prompt = PromptTemplate.from_template(
    "Classify the sentiment (positive, negative, neutral) of this message:\n\n{input}"
)
sentiment_chain = sentiment_prompt | model | parser

response_prompt = PromptTemplate.from_template(
    "Given the sentiment '{sentiment}', write a short empathetic reply to the user."
)

chain = sentiment_chain | \
        RunnableLambda(lambda sentiment: {"sentiment": sentiment}) | \
        response_prompt | \
        model | \
        parser

user_message = "I failed my interview and feel like I’ll never get a job."
result = chain.invoke({"input": user_message})
print(result)

#Examole 2

# def add_one(x: int) -> int:
#     return x + 1

# def mul_two(x: int) -> int:
#     return x * 2

# runnable_1 = RunnableLambda(add_one)
# runnable_2 = RunnableLambda(mul_two)
# sequence = runnable_1 | runnable_2
# # Or equivalently:
# # sequence = RunnableSequence(first=runnable_1, last=runnable_2)
# print(sequence.invoke(1))