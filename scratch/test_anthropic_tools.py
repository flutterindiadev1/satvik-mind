import os
import anthropic

client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])

WIKIPEDIA_TOOL_SCHEMA = {
    "name": "search_wikipedia",
    "description": "Searches Wikipedia",
    "input_schema": {
        "type": "object",
        "properties": {
            "query": {"type": "string"}
        },
        "required": ["query"]
    }
}

messages = [{"role": "user", "content": "What is the capital of France? Use the search_wikipedia tool to find out."}]
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=1500,
    messages=messages,
    tools=[WIKIPEDIA_TOOL_SCHEMA]
)

print(f"Stop reason: {response.stop_reason}")
if response.stop_reason == "tool_use":
    messages.append({"role": "assistant", "content": response.content})
    print("Appended assistant message.")
    print("Executing tool...")
    
    tool_results = []
    for block in response.content:
        if block.type == "tool_use":
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block.id,
                "content": "Paris"
            })
    
    messages.append({"role": "user", "content": tool_results})
    print("Calling again...")
    response2 = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=1500,
        messages=messages,
        tools=[WIKIPEDIA_TOOL_SCHEMA]
    )
    print(f"Final response: {response2.content[0].text}")
