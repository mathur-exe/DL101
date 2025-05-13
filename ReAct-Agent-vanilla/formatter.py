"""ReAct agent formatter for chat messages."""

from typing import List, Optional, Sequence
from .react_types import ChatMessage, Tool


DEFAULT_SYSTEM_HEADER = """You are designed to help with a variety of tasks, from answering questions to providing summaries to other types of analyses.

## Tools

You have access to a wide variety of tools. You are responsible for using the tools in any sequence you deem appropriate to complete the task at hand.
This may require breaking the task into subtasks and using different tools to complete each subtask.

You have access to the following tools:
{tool_desc}
{context_prompt}

## Output Format

Please answer in the same language as the question and use the following format:

```
Thought: The current language of the user is: (user's language). I need to use a tool to help me answer the question.
Action: tool name (one of {tool_names}) if using a tool.
Action Input: the input to the tool, in a JSON format representing the kwargs (e.g. {{"input": "hello world", "num_beams": 5}})
```

Please ALWAYS start with a Thought.

NEVER surround your response with markdown code markers. You may use code markers within your response if you need to.

Please use a valid JSON format for the Action Input. Do NOT do this {{'input': 'hello world', 'num_beams': 5}}.

If this format is used, the tool will respond in the following format:

```
Observation: tool response
```

You should keep repeating the above format till you have enough information to answer the question without using any more tools. At that point, you MUST respond in one of the following two formats:

```
Thought: I can answer without using any more tools. I'll use the user's language to answer
Answer: [your answer here (In the same language as the user's question)]
```

```
Thought: I cannot answer the question with the provided tools.
Answer: [your answer here (In the same language as the user's question)]
```

## Current Conversation

Below is the current conversation consisting of interleaving human and assistant messages.
"""

CONTEXT_SYSTEM_HEADER = DEFAULT_SYSTEM_HEADER.replace(
    "{context_prompt}",
    """
Here is some context to help you answer the question and plan:
{context}
""",
    1,
)


def get_tool_descriptions(tools: Sequence[Tool]) -> List[str]:
    """Get formatted descriptions of tools."""
    tool_descs = []
    for tool in tools:
        tool_desc = (
            f"> Tool Name: {tool.name}\n"
            f"Tool Description: {tool.description}\n"
            f"Tool Args: {tool.schema}\n"
        )
        tool_descs.append(tool_desc)
    return tool_descs


class ReActChatFormatter:
    """Formatter for chat messages in the ReAct agent."""
    
    def __init__(
        self, 
        system_header: str = DEFAULT_SYSTEM_HEADER, 
        context: str = "",
        observation_role: str = "user"
    ):
        self.system_header = system_header
        self.context = context
        self.observation_role = observation_role
    
    def format(
        self,
        tools: Sequence[Tool],
        chat_history: List[ChatMessage],
        current_reasoning: Optional[List] = None,
    ) -> List[ChatMessage]:
        """Format chat history and tools into a list of chat messages."""
        current_reasoning = current_reasoning or []
        
        format_args = {
            "tool_desc": "\n".join(get_tool_descriptions(tools)),
            "tool_names": ", ".join([tool.name for tool in tools]),
        }
        
        if self.context:
            format_args["context"] = self.context
        
        fmt_sys_header = self.system_header.format(**format_args)
        
        # Format the reasoning history as alternating user and assistant messages
        reasoning_history = []
        from .react_types import ObservationReasoningStep
        
        for reasoning_step in current_reasoning:
            if isinstance(reasoning_step, ObservationReasoningStep):
                message = ChatMessage(
                    role=self.observation_role,
                    content=reasoning_step.get_content(),
                )
            else:
                message = ChatMessage(
                    role="assistant",
                    content=reasoning_step.get_content(),
                )
            reasoning_history.append(message)
        
        return [
            ChatMessage(role="system", content=fmt_sys_header),
            *chat_history,
            *reasoning_history,
        ]
    
    @classmethod
    def from_defaults(
        cls,
        system_header: Optional[str] = None,
        context: Optional[str] = None,
        observation_role: str = "user",
    ) -> "ReActChatFormatter":
        """Create ReActChatFormatter from defaults."""
        if not system_header:
            system_header = (
                DEFAULT_SYSTEM_HEADER
                if not context
                else CONTEXT_SYSTEM_HEADER
            )
        
        return ReActChatFormatter(
            system_header=system_header,
            context=context or "",
            observation_role=observation_role,
        ) 