import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

tools = [{
    "name": "get_weather",
    "description": "Get current weather for a city. Returns temperature in celsius, conditions, and humidity. Use when the user asks about weather or temperature.",
    "input_schema": {
        "type": "object",
        "properties": {
            "city": {"type": "string", "description": "City name, e.g. 'Sydney', 'London'"}
        },
        "required": ["city"]
    }
}]

# STEP 1: Send request with tools
response = client.messages.create(
    model="claude-haiku-4-5", max_tokens=1024, tools=tools,
    messages=[{"role": "user", "content": "What's the weather like in Sydney?"}]
)

# CHECKPOINT 1: stop_reason
print(f"1. stop_reason: {response.stop_reason}")
assert response.stop_reason == "tool_use", f"Expected 'tool_use', got '{response.stop_reason}'"

# CHECKPOINT 2: Find tool_use block
tool_block = next((b for b in response.content if b.type == "tool_use"), None)
assert tool_block is not None, "No tool_use block found"
print(f"2. Tool: {tool_block.name}, ID: {tool_block.id}, Input: {tool_block.input}")

# STEP 3: Execute tool (YOUR code)
weather_data = {"temperature": 22, "conditions": "Partly cloudy", "humidity": 65}
print(f"3. Executed tool. Result: {weather_data}")

# STEP 4: Send result back
final = client.messages.create(
    model="claude-haiku-4-5", max_tokens=1024, tools=tools,
    messages=[
        {"role": "user", "content": "What's the weather like in Sydney?"},
        {"role": "assistant", "content": response.content},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tool_block.id, "content": json.dumps(weather_data)}]}
    ]
)

# CHECKPOINT 5: Final response
print(f"4. Final stop_reason: {final.stop_reason}")
assert final.stop_reason == "end_turn", f"Expected 'end_turn', got '{final.stop_reason}'"
print(f"5. Response: {final.content[0].text}")