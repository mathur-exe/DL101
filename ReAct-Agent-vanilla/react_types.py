"""Base types for ReAct agent."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional, Any

class BaseReasoningStep(ABC):
    """Base class for reasoning steps in the ReAct agent."""
    
    @abstractmethod
    def get_content(self) -> str:
        """Get the content of the reasoning step."""
        pass
    
    @property
    @abstractmethod
    def is_done(self) -> bool:
        """Indicates if this reasoning step is the final one."""
        pass


@dataclass
class ActionReasoningStep(BaseReasoningStep):
    """A reasoning step that represents an action to be taken."""
    
    thought: str
    action: str
    action_input: Dict[str, Any]
    
    def get_content(self) -> str:
        """Get formatted content for the action step."""
        return (
            f"Thought: {self.thought}\nAction: {self.action}\n"
            f"Action Input: {self.action_input}"
        )
    
    @property
    def is_done(self) -> bool:
        """An action step is not the final step."""
        return False


@dataclass
class ObservationReasoningStep(BaseReasoningStep):
    """A reasoning step that represents an observation (result of an action)."""
    
    observation: str
    return_direct: bool = False
    
    def get_content(self) -> str:
        """Get formatted content for the observation step."""
        return f"Observation: {self.observation}"
    
    @property
    def is_done(self) -> bool:
        """An observation step is final only if it should be returned directly."""
        return self.return_direct


@dataclass
class ResponseReasoningStep(BaseReasoningStep):
    """A reasoning step that represents a final response."""
    
    thought: str
    response: str
    is_streaming: bool = False
    
    def get_content(self) -> str:
        """Get formatted content for the response step."""
        if self.is_streaming:
            return f"Thought: {self.thought}\nAnswer (Starts With): {self.response} ..."
        else:
            return f"Thought: {self.thought}\nAnswer: {self.response}"
    
    @property
    def is_done(self) -> bool:
        """A response step is always the final step."""
        return True


@dataclass
class ChatMessage:
    """Represents a chat message."""
    
    content: str
    role: str  # 'system', 'user', 'assistant', 'tool'


@dataclass
class Tool:
    """Represents a tool that the agent can use."""
    
    name: str
    description: str
    function: callable
    schema: Dict[str, Any]  # JSON schema for the tool's arguments
    
    def __call__(self, **kwargs):
        """Execute the tool with the given arguments."""
        return self.function(**kwargs) 