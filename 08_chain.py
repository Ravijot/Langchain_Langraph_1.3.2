from langchain.chat_models import init_chat_model
from dotenv import load_dotenv
from langchain_core.prompts import  PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.output_parsers import StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()

model = init_chat_model(model="gpt-5.4")

prompt = PromptTemplate.from_template(template="{input}", input=["input"])

chain1 = prompt | model 

print(chain1.invoke({"input":"Hello"}))

chain2 = prompt | model | StrOutputParser()

print(chain2.invoke({"input":"hello"}))


class ResponseFormatter(BaseModel):
    emotion : str = Field(description="The emotion conveyed in the response e.g. happy, sad, angry, etc.")
    confidence_score : float = Field(description="The confidence score of the emotion conveyed in the response, between 0 and 1.")

parser = PydanticOutputParser(pydantic_object=ResponseFormatter)

emotion_prompt = prompt = PromptTemplate(
    input_variables=["text"],
    template="What emotion is conveyed in the following text?\n\n{text}\n\n{format_instructions}",
    partial_variables={"format_instructions": parser.get_format_instructions()}
)

chain3 = emotion_prompt | model | parser
response = chain3.invoke({"text": "Hurrah! I just won the lottery!"})
print(response)s