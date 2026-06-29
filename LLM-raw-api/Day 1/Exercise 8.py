import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
tools = [{"name": "get_weather", "description": "Get weather for a city.", "input_schema": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}]

question = "What is the capital of France?"  # No tool needed for this

# AUTO: Claude decides
r1 = client.messages.create(model="claude-haiku-4-5", max_tokens=256, tools=tools, tool_choice={"type": "auto"}, messages=[{"role": "user", "content": question}])
print(f"AUTO:     stop_reason={r1.stop_reason}")

# ANY: Forced to use a tool
r2 = client.messages.create(model="claude-haiku-4-5", max_tokens=256, tools=tools, tool_choice={"type": "any"}, messages=[{"role": "user", "content": question}])
print(f"ANY:      stop_reason={r2.stop_reason}")
if r2.stop_reason == "tool_use":
    tb = next(b for b in r2.content if b.type == "tool_use")
    print(f"          forced: {tb.name}({tb.input})")

# SPECIFIC: Forced to use get_weather
r3 = client.messages.create(model="claude-haiku-4-5", max_tokens=256, tools=tools, tool_choice={"type": "tool", "name": "get_weather"}, messages=[{"role": "user", "content": question}])
print(f"SPECIFIC: stop_reason={r3.stop_reason}")
if r3.stop_reason == "tool_use":
    tb = next(b for b in r3.content if b.type == "tool_use")
    print(f"          forced: {tb.name}({tb.input})")