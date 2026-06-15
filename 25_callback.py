from langchain_core.callbacks import BaseCallbackHandler
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

llm = init_chat_model(model="gpt-5.4")

class MyHandler(BaseCallbackHandler):

    def on_llm_start(self, serialized, prompts, **kwargs):
        print("\n=== LLM START ===")
        print("Prompt:")
        for prompt in prompts:
            print(prompt)
        print()

    def on_llm_end(self, response, **kwargs):
        print("\n\n=== LLM END ===")

        # Full generated text
        if response.generations:
            text = response.generations[0][0].text
            print(f"Response:\n{text}")

        # Model info (depends on provider)
        if hasattr(response, "llm_output") and response.llm_output:
            print("\nLLM Output:")
            print(response.llm_output)

        print("=================\n")


response = llm.invoke(
    "Tell me a joke",
    config={
        "callbacks": [MyHandler()]
    }
)

print(response)