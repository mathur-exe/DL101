from llm_client import llm_client
from open_router import open_router

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
# response = llm_client("microsoft/phi-4", messages)
# print(response)

response = open_router("google/gemma-3n-e4b-it:free", messages)
print(response)