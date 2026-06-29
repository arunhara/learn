import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

tools = [
    {"name": "search_employees", "description": "Search for employees by name, department, or office. Use when the user wants to FIND an employee. Do NOT use for department statistics.",
     "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_department_stats", "description": "Get aggregate statistics for a department: headcount, attrition rate, avg tenure. Use for department-level METRICS. Do NOT use for individual employee lookups.",
     "input_schema": {"type": "object", "properties": {"department": {"type": "string"}}, "required": ["department"]}},
    {"name": "get_policy_info", "description": "Look up HR policies on topics like leave, benefits, remote work. Use for POLICY questions. Do NOT use for employee data or department metrics.",
     "input_schema": {"type": "object", "properties": {"topic": {"type": "string"}}, "required": ["topic"]}}
]

tests = [
    ("Who is Sarah Chen?", "search_employees"),
    ("How many people are in Engineering?", "get_department_stats"),
    ("What is our parental leave policy?", "get_policy_info"),
    ("Find employees in the Sydney office", "search_employees"),
    ("What's the attrition rate for Sales?", "get_department_stats"),
    ("What are the rules for working from home?", "get_policy_info"),
]

correct = 0
for query, expected in tests:
    r = client.messages.create(model="claude-haiku-4-5", max_tokens=512, tools=tools, messages=[{"role": "user", "content": query}])
    if r.stop_reason == "tool_use":
        actual = next(b.name for b in r.content if b.type == "tool_use")
        match = "✅" if actual == expected else "❌"
        if actual == expected: correct += 1
        print(f"{match} '{query}' → expected: {expected} | got: {actual}")
    else:
        print(f"❌ '{query}' → no tool call (stop_reason: {r.stop_reason})")

print(f"\nScore: {correct}/{len(tests)}")
