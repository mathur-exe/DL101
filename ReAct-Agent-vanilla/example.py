"""Example usage of the ReAct agent."""

import os
import json
import datetime
import requests
from typing import Dict, Any, List

from react_agent import ReActAgent, Tool


def weather_tool(location: str) -> str:
    """Get the current weather for a location."""
    # This is a mock implementation
    return f"Weather in {location}: 72°F, Partly Cloudy"


def search_tool(query: str) -> str:
    """Search the web for information."""
    # This is a mock implementation
    return f"Search results for '{query}':\n1. Relevant information about {query}\n2. More details about {query}"


def calculator_tool(expression: str) -> str:
    """Calculate a mathematical expression."""
    try:
        # Be very careful with eval - in a real implementation you'd want to use a safer alternative
        result = eval(expression, {"__builtins__": {}})
        return f"Result: {result}"
    except Exception as e:
        return f"Error calculating: {str(e)}"


def current_time_tool() -> str:
    """Get the current time."""
    now = datetime.datetime.now()
    return f"Current time: {now.strftime('%Y-%m-%d %H:%M:%S')}"


def main():
    """Run an example ReAct agent session."""
    # Define the tools
    tools = [
        Tool(
            name="weather",
            description="Get the current weather for a location",
            function=weather_tool,
            schema={"location": "string"}
        ),
        Tool(
            name="search",
            description="Search the web for information",
            function=search_tool,
            schema={"query": "string"}
        ),
        Tool(
            name="calculator",
            description="Calculate a mathematical expression",
            function=calculator_tool,
            schema={"expression": "string"}
        ),
        Tool(
            name="current_time",
            description="Get the current time",
            function=current_time_tool,
            schema={}
        ),
    ]
    
    # Create the agent
    agent = ReActAgent(
        tools=tools,
        model="alpha-42",  # replace with the actual model name if different
        max_iterations=10,
        verbose=True
    )
    
    # Run an example query
    query = input("Enter your query: ")
    
    print("\n===== RUNNING REACT AGENT =====")
    response = agent.run(query)
    
    print("\n===== FINAL RESPONSE =====")
    print(response)


if __name__ == "__main__":
    main() 