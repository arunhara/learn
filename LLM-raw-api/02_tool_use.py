import os
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2A — Define a tool and observe the tool_use block
# Goal: see what the model actually returns when it decides to call a tool
# ─────────────────────────────────────────────────────────────────────────────

tools = [
    {
        "name": "get_current_time",
        "description": "Returns the current date and time as a string.",
        "input_schema": {
            "type": "object",
            "properties": {},
            "required": []
        }
    }
]

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    tools=tools,
    messages=[{"role": "user", "content": "What time is it right now?"}]
)

print("=== EXERCISE 2A: Tool Use Response ===")
print(f"stop_reason: {response.stop_reason}")   # should be "tool_use"
print(f"content blocks: {len(response.content)}")
for i, block in enumerate(response.content):
    print(f"\nBlock {i}: type={block.type}")
    if block.type == "tool_use":
        print(f"  id:    {block.id}")
        print(f"  name:  {block.name}")
        print(f"  input: {block.input}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2B — The full tool use cycle (one round trip)
# Goal: send tool result back and get a final response
# ─────────────────────────────────────────────────────────────────────────────

import datetime

def get_current_time() -> str:
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

# Step 1: Initial request — model responds with tool_use
messages = [{"role": "user", "content": "What time is it right now?"}]

response1 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    tools=tools,
    messages=messages
)

# Step 2: Append the assistant's response to history (REQUIRED)
messages.append({"role": "assistant", "content": response1.content})

# Step 3: Execute the tool and build the result message
tool_results = []
for block in response1.content:
    if block.type == "tool_use":
        result = get_current_time()   # actually run the function
        print(f"Executed tool: {block.name} → {result}")
        tool_results.append({
            "type": "tool_result",
            "tool_use_id": block.id,   # must match the id from the tool_use block
            "content": result
        })

# Step 4: Append tool results as a user message
messages.append({"role": "user", "content": tool_results})

# Step 5: Call the API again — now the model has the result
response2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    tools=tools,
    messages=messages
)

print("\n=== EXERCISE 2B: Full Tool Cycle ===")
print(f"Final stop_reason: {response2.stop_reason}")  # should be "end_turn"
print(f"Final answer: {response2.content[0].text}")
print(f"Messages array length after cycle: {len(messages)}")
print()

# Print the full messages array so you can see how it accumulated
print("=== Full messages array ===")
for i, msg in enumerate(messages):
    print(f"\n[{i}] role: {msg['role']}")
    content = msg['content']
    if isinstance(content, str):
        print(f"    content: {content[:80]}")
    elif isinstance(content, list):
        for block in content:
            if hasattr(block, 'type'):
                print(f"    block type: {block.type}")
            elif isinstance(block, dict):
                print(f"    block type: {block.get('type')} | {str(block)[:80]}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2C — What happens with a wrong tool_use_id
# Goal: understand why the id field matters
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 2C: Wrong tool_use_id ===")
try:
    messages_bad = [{"role": "user", "content": "What time is it?"}]
    r = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=tools,
        messages=messages_bad
    )
    messages_bad.append({"role": "assistant", "content": r.content})
    messages_bad.append({
        "role": "user",
        "content": [{
            "type": "tool_result",
            "tool_use_id": "WRONG_ID_123",   # intentionally wrong
            "content": "12:00:00"
        }]
    })
    client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        tools=tools,
        messages=messages_bad
    )
except Exception as e:
    print(f"Error (expected): {type(e).__name__}: {e}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 2D — Tool with arguments
# Goal: define a tool that takes arguments and observe input schema enforcement
# ─────────────────────────────────────────────────────────────────────────────

import sqlite3

def run_sql(query: str) -> str:
    """Run a SQL query against an in-memory SQLite database."""
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT, created_at TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'alice', '2024-01-01')")
    conn.execute("INSERT INTO users VALUES (2, 'bob',   '2024-01-15')")
    conn.execute("INSERT INTO users VALUES (3, 'carol', '2024-02-01')")
    try:
        cursor = conn.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return "Query returned 0 rows."
        return "\n".join(str(row) for row in rows)
    except Exception as e:
        return f"ERROR: {e}"

sql_tools = [
    {
        "name": "run_sql",
        "description": "Execute a SQL SELECT query against the users table and return the results.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A valid SQL SELECT statement. Only SELECT is allowed."
                }
            },
            "required": ["query"]
        }
    }
]

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    tools=sql_tools,
    messages=[{"role": "user", "content": "How many users are in the database?"}]
)

print("=== EXERCISE 2D: Tool with Arguments ===")
for block in response.content:
    if block.type == "tool_use":
        print(f"Model generated SQL: {block.input['query']}")
        result = run_sql(block.input["query"])
        print(f"Query result: {result}")