import anthropic
import json
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()

tools = [
    {"name": "search_employees", "description": "Search employees by name, department, or office.", "input_schema": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
    {"name": "get_employee_details", "description": "Get full details for a specific employee by ID.", "input_schema": {"type": "object", "properties": {"employee_id": {"type": "string"}}, "required": ["employee_id"]}},
    {"name": "get_department_stats", "description": "Get department statistics: headcount, attrition rate.", "input_schema": {"type": "object", "properties": {"department": {"type": "string"}}, "required": ["department"]}}
]

def execute_tool(name, params):
    if name == "search_employees":
        return [{"id": "EMP-001", "name": "Sarah Chen", "department": "Data Engineering", "office": "Sydney"}]
    elif name == "get_employee_details":
        return {"id": params["employee_id"], "name": "Sarah Chen", "title": "Senior Data Engineer", "department": "Data Engineering", "manager": "Arun Kumar"}
    elif name == "get_department_stats":
        return {"department": params["department"], "headcount": 12, "attrition_rate": "8%", "open_positions": 2}
    return {"error": f"Unknown tool: {name}"}

task = "Find who works in Data Engineering in Sydney, get details on the first person, and tell me the department's attrition rate."
messages = [{"role": "user", "content": task}]
tool_sequence = []

print(f"TASK: {task}\n")

for iteration in range(10):
    response = client.messages.create(
        model="claude-haiku-4-5", max_tokens=2048, tools=tools,
        system="You are a workforce analytics assistant.",
        messages=messages
    )
    print(f"--- Iteration {iteration + 1} | stop_reason: {response.stop_reason} ---")

    if response.stop_reason == "end_turn":
        final = next((b.text for b in response.content if b.type == "text"), "Done.")
        print(f"\nFINAL: {final}")
        break

    if response.stop_reason == "tool_use":
        messages.append({"role": "assistant", "content": response.content})
        results = []
        for block in response.content:
            if block.type == "tool_use":
                print(f"  → {block.name}({json.dumps(block.input)})")
                tool_sequence.append(block.name)
                result = execute_tool(block.name, block.input)
                results.append({"type": "tool_result", "tool_use_id": block.id, "content": json.dumps(result)})
        messages.append({"role": "user", "content": results})

print(f"\nTool sequence: {tool_sequence}")
print(f"Total iterations: {iteration + 1}")
print(f"Messages in array: {len(messages)}")