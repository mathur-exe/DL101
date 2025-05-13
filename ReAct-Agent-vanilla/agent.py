"""ReAct agent implementation in vanilla Python."""

import json
from typing import Dict, List, Optional, Sequence, Any, Tuple, Union
import openai

from .react_types import (
    ActionReasoningStep,
    BaseReasoningStep,
    ChatMessage,
    ObservationReasoningStep,
    ResponseReasoningStep,
    Tool
)
from .formatter import ReActChatFormatter
from .output_parser import ReActOutputParser


class ReActAgent:
    """
    ReAct (Reasoning and Acting) agent implementation.
    
    This agent uses the ReAct framework to decide on which tools to use
    and in what order to solve a given task.
    """
    
    def __init__(
        self,
        tools: Sequence[Tool],
        model: str = "alpha-42",
        max_iterations: int = 10,
        chat_formatter: Optional[ReActChatFormatter] = None,
        output_parser: Optional[ReActOutputParser] = None,
        verbose: bool = False,
        context: Optional[str] = None,
    ):
        """
        Initialize the ReAct agent.
        
        Args:
            tools: List of tools the agent can use
            model: Name of the LLM model to use
            max_iterations: Maximum number of reasoning iterations
            chat_formatter: Formatter for chat messages
            output_parser: Parser for model outputs
            verbose: Whether to print detailed information during execution
            context: Optional context to provide to the agent
        """
        self.tools = {tool.name: tool for tool in tools}
        self.model = model
        self.max_iterations = max_iterations
        self.verbose = verbose
        
        # Initialize chat formatter with context if provided
        if context and chat_formatter:
            raise ValueError("Cannot provide both context and chat_formatter")
        if context:
            self.chat_formatter = ReActChatFormatter.from_defaults(context=context)
        else:
            self.chat_formatter = chat_formatter or ReActChatFormatter()
            
        self.output_parser = output_parser or ReActOutputParser()
        self.chat_history = []
        
    def reset(self):
        """Reset the agent state."""
        self.chat_history = []
    
    def add_message(self, message: str, role: str = "user"):
        """
        Add a message to the chat history.
        
        Args:
            message: The content of the message
            role: The role of the sender ('user', 'assistant', or 'system')
        """
        self.chat_history.append(ChatMessage(content=message, role=role))
    
    def _get_llm_response(self, messages: List[Dict[str, str]]) -> str:
        """
        Get a response from the language model.
        
        Args:
            messages: List of message dictionaries for the model
            
        Returns:
            The model's response text
        """
        response = openai.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,  # Deterministic responses for predictable behavior
        )
        return response.choices[0].message.content
    
    def _format_messages_for_llm(
        self, 
        chat_history: List[ChatMessage], 
        current_reasoning: List[BaseReasoningStep]
    ) -> List[Dict[str, str]]:
        """
        Format messages for the LLM API.
        
        Args:
            chat_history: List of chat messages
            current_reasoning: List of reasoning steps
            
        Returns:
            List of message dictionaries for the model
        """
        formatted_messages = self.chat_formatter.format(
            tools=list(self.tools.values()),
            chat_history=chat_history,
            current_reasoning=current_reasoning,
        )
        
        # Convert to the format expected by the LLM API
        return [
            {"role": msg.role, "content": msg.content}
            for msg in formatted_messages
        ]
    
    def _execute_tool(
        self, 
        tool_name: str, 
        tool_args: Dict[str, Any]
    ) -> Tuple[str, bool]:
        """
        Execute a tool with the given arguments.
        
        Args:
            tool_name: Name of the tool to execute
            tool_args: Arguments for the tool
            
        Returns:
            Tuple of (result, should_return_direct)
            
        Raises:
            ValueError: If the tool is not found
        """
        if tool_name not in self.tools:
            available_tools = ", ".join(self.tools.keys())
            error_msg = f"Tool '{tool_name}' not found. Available tools: {available_tools}"
            if self.verbose:
                print(f"Tool error: {error_msg}")
            return error_msg, False
        
        try:
            tool = self.tools[tool_name]
            result = tool(**tool_args)
            return str(result), False
        except Exception as e:
            error_msg = f"Error executing tool '{tool_name}': {str(e)}"
            if self.verbose:
                print(f"Tool execution error: {error_msg}")
            return error_msg, False
    
    def _run_iteration(
        self, 
        current_reasoning: List[BaseReasoningStep]
    ) -> Tuple[BaseReasoningStep, bool]:
        """
        Run a single iteration of the reasoning process.
        
        Args:
            current_reasoning: List of reasoning steps so far
            
        Returns:
            Tuple of (new_reasoning_step, is_done)
        """
        # Format messages for the LLM
        messages = self._format_messages_for_llm(
            chat_history=self.chat_history,
            current_reasoning=current_reasoning,
        )
        
        if self.verbose:
            print("\n===== SENDING TO LLM =====")
            for msg in messages:
                print(f"[{msg['role']}]: {msg['content'][:100]}...")
        
        # Get a response from the LLM
        response = self._get_llm_response(messages)
        
        if self.verbose:
            print("\n===== LLM RESPONSE =====")
            print(response)
        
        # Parse the response
        try:
            reasoning_step = self.output_parser.parse(response)
            
            # If the reasoning step is a final response, we're done
            if reasoning_step.is_done:
                return reasoning_step, True
            
            # If it's an action step, execute the tool
            if isinstance(reasoning_step, ActionReasoningStep):
                tool_name = reasoning_step.action
                tool_args = reasoning_step.action_input
                
                # Execute the tool
                result, return_direct = self._execute_tool(tool_name, tool_args)
                
                # Create an observation step
                observation_step = ObservationReasoningStep(
                    observation=result,
                    return_direct=return_direct,
                )
                
                return observation_step, return_direct
            
            # Shouldn't reach here if the output parser is working correctly
            raise ValueError(f"Unexpected reasoning step type: {type(reasoning_step)}")
            
        except ValueError as e:
            if self.verbose:
                print(f"Error parsing LLM output: {str(e)}")
            
            # Create an observation step with the error message
            error_msg = "Error: I couldn't understand the output format. Please follow the correct format."
            observation_step = ObservationReasoningStep(
                observation=error_msg,
                return_direct=False,
            )
            
            return observation_step, False
    
    def run(self, query: str) -> str:
        """
        Run the ReAct agent on a query.
        
        Args:
            query: The user's query
            
        Returns:
            The agent's final response
        """
        # Reset and add the query as the first message
        self.reset()
        self.add_message(query)
        
        # Initialize reasoning steps
        current_reasoning = []
        
        # Run iterations until we get a final response or reach max iterations
        for iteration in range(self.max_iterations):
            if self.verbose:
                print(f"\n===== ITERATION {iteration + 1} =====")
            
            # Run a single iteration
            reasoning_step, is_done = self._run_iteration(current_reasoning)
            
            # Add the reasoning step to the list
            current_reasoning.append(reasoning_step)
            
            # If we're done, extract the response
            if is_done and isinstance(reasoning_step, ResponseReasoningStep):
                return reasoning_step.response
                
            # If we're done but don't have a response, something went wrong
            elif is_done:
                return "I'm not sure how to answer that."
        
        # If we reached max iterations without getting a final response
        return "I'm still thinking about this. Let me try a different approach next time." 