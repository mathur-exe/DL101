from typing import Dict, List, Literal

import os
from openai import OpenAI, base_url

class LLM_Client:
    def __init__(self, 
    base_url: str, 
    api_key: str, 
    model_name: str) -> None:
        self.base_url : str = base_url
        self.api_key : str = api_key
        self.model_name : str = model_name
        self.client = OpenAI(
            base_url=self.base_url,
            api_key=self.api_key,
        )
    
    def chat_completion(self, 
        messages: List[Dict[str, str]],
        model_name : str = None,
        reasoning : Dict[bool, str] = None,
        return_type : Literal["text", "api_response"] = "text"
    ):
        inf_model_name = model_name if model_name is not None else self.model_name
        reasoning_effort = reasoning[True] if reasoning is not None else None
        
        completion = self.client.chat.completions.create(
            model=inf_model_name,
            messages=messages,
            reasoning_effort=reasoning_effort,
        )

        resp = completion.choices[0].message.content
        messages.append({"role" : "assistant", "content" : resp})
        return resp, messages

    def self_reflect(self, 
        reflect_prompt : str,
        messages : List[Dict[str, str]],
        n_iter : int = 3
    ) -> str:
        '''
        Given a reflection prompt, the llm must call `chat_completion` for n-iter 
        Args:
            reflect_prompt : str
            messages : List[Dict[str, str]]
            n_iter : int
        '''
        for i in range(n_iter):
            messages.append({"role" : "user", "content" : reflect_prompt})
            resp, messages = self.chat_completion(messages)
            messages.append({"role" : "assistant", "content" : resp})
            print(f"Iteration {i+1}: {resp}")

        return messages