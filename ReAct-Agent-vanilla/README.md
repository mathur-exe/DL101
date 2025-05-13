# ReAct Agent in Vanilla Python

This is a vanilla Python implementation of the ReAct (Reasoning and Acting) agent, inspired by the LlamaIndex implementation. The ReAct framework enables language models to solve tasks by interleaving reasoning and actions.

## Features

- 🤔 **Reasoning**: Generates reasoning about how to approach a task
- 🛠️ **Action**: Executes tools based on its reasoning
- 👁️ **Observation**: Observes the results of actions
- 🔄 **Iteration**: Iterates through reasoning-action-observation until it reaches a final answer

## Installation

Clone this repository and install the required packages:

```bash
# No special installation required beyond openai package
pip install openai
```

## Usage

```python
from react_agent import ReActAgent, Tool

# Define tools that the agent can use
tools = [
    Tool(
        name="calculator",
        description="Calculate a mathematical expression",
        function=lambda expression: str(eval(expression, {"__builtins__": {}})),
        schema={"expression": "string"}
    ),
    # Define more tools...
]

# Create the agent
agent = ReActAgent(
    tools=tools,
    model="alpha-42",  # Your model name
    max_iterations=10,
    verbose=True
)

# Run a query
response = agent.run("What is the square root of 256?")
print(response)
```

See `example.py` for a complete example with multiple tools.

## Architecture

The implementation consists of several key components:

1. **Types**: Core data structures for representing reasoning steps, tools, and messages
2. **Formatter**: Formats prompts and chat history for the LLM
3. **Output Parser**: Parses LLM outputs into structured reasoning steps
4. **Agent**: Orchestrates the reasoning process and tool executions

## Customization

You can customize the agent by:

- Creating your own tools
- Customizing the prompt templates
- Adjusting the maximum number of iterations
- Providing additional context
- Using a different LLM

## Notes

This implementation assumes a vLLM-deployed model with an OpenAI-compatible API.

## License

This project is available under the MIT License. 