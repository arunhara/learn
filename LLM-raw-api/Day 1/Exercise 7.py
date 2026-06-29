import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

tools = [{"name": "get_employee", "description": "Get employee details by ID.", "input_schema": {"type": "object", "properties": {"id": {"type": "string"}}, "required": ["id"]}}]

# Get a tool call
r = client.messages.create(model="claude-haiku-4-5", max_tokens=512, tools=tools, messages=[{"role": "user", "content": "Look up employee EMP-99999"}])
tb = next(b for b in r.content if b.type == "tool_use")

# WITH is_error flag
r1 = client.messages.create(model="claude-haiku-4-5", max_tokens=512, tools=tools, messages=[
    {"role": "user", "content": "Look up employee EMP-99999"},
    {"role": "assistant", "content": r.content},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tb.id, "is_error": True, "content": "Error: Employee EMP-99999 not found."}]}
])
print("WITH is_error=True:")
print(f"  {r1.content[0].text}\n")

# WITHOUT is_error flag (same error message, no flag)
r2 = client.messages.create(model="claude-haiku-4-5", max_tokens=512, tools=tools, messages=[
    {"role": "user", "content": "Look up employee EMP-99999"},
    {"role": "assistant", "content": r.content},
    {"role": "user", "content": [{"type": "tool_result", "tool_use_id": tb.id, "content": "Error: Employee EMP-99999 not found."}]}
])
print("WITHOUT is_error flag:")
print(f"  {r2.content[0].text}")