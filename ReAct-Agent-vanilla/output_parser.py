"""Output parser for ReAct agent."""

import re
import json
from typing import Tuple, Dict, Any

from .react_types import ActionReasoningStep, ResponseReasoningStep, BaseReasoningStep


def extract_tool_use(input_text: str) -> Tuple[str, str, str]:
    """
    Extract tool use from the LLM output.
    
    Args:
        input_text: The text output from the LLM
        
    Returns:
        Tuple of (thought, action, action_input)
        
    Raises:
        ValueError: If the input text doesn't match the expected format
    """
    pattern = (
        r"\s*Thought: (.*?)\n+Action: ([^\n\(\) ]+).*?\n+Action Input: .*?(\{.*\})"
    )

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(f"Could not extract tool use from input text: {input_text}")

    thought = match.group(1).strip()
    action = match.group(2).strip()
    action_input = match.group(3).strip()
    return thought, action, action_input


def extract_final_response(input_text: str) -> Tuple[str, str]:
    """
    Extract the final answer from the LLM output.
    
    Args:
        input_text: The text output from the LLM
        
    Returns:
        Tuple of (thought, answer)
        
    Raises:
        ValueError: If the input text doesn't match the expected format
    """
    pattern = r"\s*Thought:(.*?)Answer:(.*?)(?:$)"

    match = re.search(pattern, input_text, re.DOTALL)
    if not match:
        raise ValueError(
            f"Could not extract final answer from input text: {input_text}"
        )

    thought = match.group(1).strip()
    answer = match.group(2).strip()
    return thought, answer


def parse_action_input(action_input_str: str) -> Dict[str, Any]:
    """
    Parse the action input string into a dictionary.
    
    Args:
        action_input_str: The action input string from the LLM
        
    Returns:
        Dictionary of action input parameters
    """
    # First try to parse using json
    try:
        return json.loads(action_input_str)
    except json.JSONDecodeError:
        # Fall back to manual parsing for simple cases
        cleaned_str = action_input_str.replace("'", '"')
        try:
            return json.loads(cleaned_str)
        except json.JSONDecodeError:
            # Very basic parsing for simpler cases
            result = {}
            # Extract key-value pairs using regex
            pattern = r'"(\w+)":\s*"([^"]*)"'
            matches = re.findall(pattern, action_input_str)
            if matches:
                return dict(matches)
            
            # If all else fails, just return the string as is
            return {"input": action_input_str}


class ReActOutputParser:
    """Parser for ReAct agent outputs."""
    
    def parse(self, output: str, is_streaming: bool = False) -> BaseReasoningStep:
        """
        Parse output from ReAct agent.
        
        Args:
            output: The text output from the LLM
            is_streaming: Whether the output is from a streaming response
            
        Returns:
            A BaseReasoningStep object (ActionReasoningStep or ResponseReasoningStep)
            
        Raises:
            ValueError: If the output cannot be parsed
        """
        if "Thought:" not in output:
            # Handle case where agent directly outputs the answer
            return ResponseReasoningStep(
                thought="(Implicit) I can answer without any more tools!",
                response=output,
                is_streaming=is_streaming,
            )
        
        # An "Action" should take priority over an "Answer"
        if "Action:" in output:
            try:
                thought, action, action_input_str = extract_tool_use(output)
                action_input_dict = parse_action_input(action_input_str)
                return ActionReasoningStep(
                    thought=thought, 
                    action=action, 
                    action_input=action_input_dict
                )
            except ValueError as e:
                raise ValueError(f"Failed to parse action: {str(e)}")
        
        if "Answer:" in output:
            try:
                thought, answer = extract_final_response(output)
                return ResponseReasoningStep(
                    thought=thought, 
                    response=answer, 
                    is_streaming=is_streaming
                )
            except ValueError as e:
                raise ValueError(f"Failed to parse response: {str(e)}")
        
        raise ValueError(f"Could not parse output: {output}") 