from llm_client import llm_client

messages = [
    {"role": "user", "content": "What is the capital of France?"}
]
response = llm_client("microsoft/phi-4", messages)
print(response)