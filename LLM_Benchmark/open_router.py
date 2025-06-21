from loadenv import loadenv
loadenv()
import os

from openai import OpenAI
def open_router(
        model: str,
        messages: list[dict],
        ):
    client = OpenAI(
        base_url=os.getenv("OPEN_ROUTER_BASE_URL"),
        api_key=os.getenv("OPEN_ROUTER"),
    )

    completion = client.chat.completions.create(
        model=model,
        messages=messages,
    )
    return completion.choices[0].message.content