from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

def llm_client(
        model_name: str,
        messages: list[dict],
):
    client = OpenAI(
        base_url=os.environ["PHI4_BASEURL"],
        api_key=os.environ["HF_TOKEN"],
    )

    completion = client.chat.completions.create(
        model=model_name,
        messages=messages,
    )

    return completion.choices[0].message.content