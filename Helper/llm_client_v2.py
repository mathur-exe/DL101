from dataclasses import dataclass
import os
from typing import Dict, List, Literal
from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI, base_url

@dataclass
class OpenRouterClient:
    api_key: str = os.getenv("OPENROUTER_API_KEY")
    api_base: str = os.getenv("OPENROUTER_BASE_URL")

from dataclasses import dataclass, field

@dataclass
class Repo:
    name: str
    contributors: list[str] = field(default_factory=list)

r1 = Repo("megacorp")
r1.contributors.append("bob")
r2 = Repo("tiny")
assert r2.contributors == []   # safe: different list instances


