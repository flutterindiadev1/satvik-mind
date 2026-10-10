import json
import wikipedia
from typing import Any, Dict, List

def search_wikipedia(query: str) -> str:
    """
    Search Wikipedia for the given query and return a summary of the top result.
    """
    try:
        results = wikipedia.search(query, results=1)
        if not results:
            return f"No Wikipedia results found for: {query}"
        
        page = wikipedia.page(results[0], auto_suggest=False)
        return f"Title: {page.title}\nSummary: {page.summary[:1500]}"
    except wikipedia.DisambiguationError as e:
        # Just grab the first option to be simple for now
        try:
            page = wikipedia.page(e.options[0], auto_suggest=False)
            return f"Title: {page.title}\nSummary: {page.summary[:1500]}"
        except Exception:
            return f"Disambiguation error for: {query}. Too many options."
    except Exception as e:
        return f"Error searching Wikipedia: {str(e)}"

# The schema structure that Anthropic expects
WIKIPEDIA_TOOL_SCHEMA = {
    "name": "search_wikipedia",
    "description": "Searches Wikipedia and returns a summary of the most relevant page. Use this to find factual information, historical events, science, or general knowledge.",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query (e.g. 'Aspirin side effects' or 'Nyaya philosophy')"
            }
        },
        "required": ["query"]
    }
}

from causal.sandbox import MedicalSandbox
_sandbox = MedicalSandbox()

def observe_sandbox(n_samples: int = 1000) -> str:
    return json.dumps(_sandbox.observe(n_samples))

def intervene_sandbox(treatment: int, n_samples: int = 1000) -> str:
    return json.dumps(_sandbox.do(treatment, n_samples))

OBSERVE_TOOL_SCHEMA = {
    "name": "observe_sandbox",
    "description": "Observes natural data from the medical sandbox to see correlations between Treatment and Recovery. Do this before intervening.",
    "input_schema": {
        "type": "object",
        "properties": {
            "n_samples": {"type": "integer", "description": "Number of patients to observe (default 1000)"}
        }
    }
}

INTERVENE_TOOL_SCHEMA = {
    "name": "intervene_sandbox",
    "description": "Forces a treatment intervention on the population (do-calculus) to measure causal effect. Treatment: 1 for Drug X, 0 for Placebo.",
    "input_schema": {
        "type": "object",
        "properties": {
            "treatment": {"type": "integer", "description": "1 to give Drug X, 0 to give Placebo"},
            "n_samples": {"type": "integer", "description": "Number of patients (default 1000)"}
        },
        "required": ["treatment"]
    }
}

ALL_TOOL_SCHEMAS = [WIKIPEDIA_TOOL_SCHEMA, OBSERVE_TOOL_SCHEMA, INTERVENE_TOOL_SCHEMA]

# The schema structure that OpenAI expects
OPENAI_TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": WIKIPEDIA_TOOL_SCHEMA["name"],
            "description": WIKIPEDIA_TOOL_SCHEMA["description"],
            "parameters": WIKIPEDIA_TOOL_SCHEMA["input_schema"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": OBSERVE_TOOL_SCHEMA["name"],
            "description": OBSERVE_TOOL_SCHEMA["description"],
            "parameters": OBSERVE_TOOL_SCHEMA["input_schema"]
        }
    },
    {
        "type": "function",
        "function": {
            "name": INTERVENE_TOOL_SCHEMA["name"],
            "description": INTERVENE_TOOL_SCHEMA["description"],
            "parameters": INTERVENE_TOOL_SCHEMA["input_schema"]
        }
    }
]

AVAILABLE_TOOLS = {
    "search_wikipedia": search_wikipedia,
    "observe_sandbox": observe_sandbox,
    "intervene_sandbox": intervene_sandbox
}

def execute_tool(name: str, input_data: Dict[str, Any]) -> str:
    """Execute a tool by name and return its output as a string."""
    if name not in AVAILABLE_TOOLS:
        return f"Error: Tool {name} not found."
    
    try:
        func = AVAILABLE_TOOLS[name]
        return func(**input_data)
    except Exception as e:
        return f"Error executing tool {name}: {e}"
