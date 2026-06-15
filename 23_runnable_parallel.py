import os 
from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableParallel
from langchain_core.runnables import RunnableLambda

load_dotenv()
    
model = init_chat_model("gpt-5.4")
parser = StrOutputParser()

prompt1 = PromptTemplate.from_template("What is the capital of {place}?")
prompt2 = PromptTemplate.from_template("What is the population of {place}?")

# Build two chains
chain1 = prompt1 | model | parser
chain2 = prompt2 | model | parser


parallel_chain = RunnableParallel({
    "capital": chain1,
    "population": chain2
})

result = parallel_chain.invoke({"place": "Japan"})
print(result)

#Example 2

# def add_one(x: int) -> int:
#     return x + 1

# def mul_two(x: int) -> int:
#     return x * 2

# def mul_three(x: int) -> int:
#     return x * 3

# runnable_1 = RunnableLambda(add_one)
# runnable_2 = RunnableLambda(mul_two)
# runnable_3 = RunnableLambda(mul_three)

# sequence = runnable_1 | {  # this dict is coerced to a RunnableParallel
#     "mul_two": runnable_2,
#     "mul_three": runnable_3,
# }
# # Or equivalently:
# # sequence = runnable_1 | RunnableParallel(
# #     {"mul_two": runnable_2, "mul_three": runnable_3}
# # )
# # Also equivalently:
# # sequence = runnable_1 | RunnableParallel(
# #     mul_two=runnable_2,
# #     mul_three=runnable_3,
# # )

# print(sequence.invoke(1))
