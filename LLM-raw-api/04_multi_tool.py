import os
import sqlite3
import json
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# ─── Extended database ────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT, email TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, total REAL, status TEXT, created_at TEXT)")
    conn.execute("CREATE TABLE products (id INTEGER, name TEXT, price REAL, category TEXT)")
    conn.execute("INSERT INTO users VALUES (1,'alice','alice@co.com','2024-01-01')")
    conn.execute("INSERT INTO users VALUES (2,'bob','bob@co.com','2024-01-15')")
    conn.execute("INSERT INTO orders VALUES (1,1,99.99,'complete','2024-01-10')")
    conn.execute("INSERT INTO orders VALUES (2,1,149.50,'complete','2024-01-20')")
    conn.execute("INSERT INTO orders VALUES (3,2,25.00,'pending','2024-02-01')")
    conn.execute("INSERT INTO products VALUES (1,'Widget A',49.99,'hardware')")
    conn.execute("INSERT INTO products VALUES (2,'Widget B',149.00,'hardware')")
    conn.commit()
    return conn

DB = get_db()

# ─── Tool implementations ─────────────────────────────────────────────────────

def get_schema() -> str:
    cursor = DB.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    parts = []
    for table in tables:
        c = DB.execute(f"PRAGMA table_info({table})")
        cols = [f"{r[1]} {r[2]}" for r in c.fetchall()]
        parts.append(f"{table}({', '.join(cols)})")
    return "\n".join(parts)

def run_sql(query: str) -> str:
    try:
        c = DB.execute(query)
        rows = c.fetchall()
        if not rows:
            return "0 rows."
        col_names = [d[0] for d in c.description]
        lines = [" | ".join(col_names)]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"

def validate_sql(query: str) -> str:
    """Check if SQL is a safe SELECT-only query."""
    q = query.strip().upper()
    if not q.startswith("SELECT"):
        return "INVALID: Only SELECT queries are permitted."
    forbidden = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER"]
    for keyword in forbidden:
        if keyword in q:
            return f"INVALID: Query contains forbidden keyword '{keyword}'."
    return "VALID"

def execute_tool(name: str, inputs: dict) -> str:
    if name == "get_schema":   return get_schema()
    if name == "run_sql":      return run_sql(inputs["query"])
    if name == "validate_sql": return validate_sql(inputs["query"])
    return f"ERROR: Unknown tool '{name}'"

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4A — Three tools: observe the selection order
# Goal: see how tool descriptions guide model choices
# ─────────────────────────────────────────────────────────────────────────────

TOOLS_V1 = [
    {
        "name": "get_schema",
        "description": "Returns table names and column definitions. ALWAYS call this first before writing any SQL query.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "validate_sql",
        "description": "Check if a SQL query is safe and permitted before running it. Call this after writing SQL but BEFORE calling run_sql.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "The SQL query to validate."}
            },
            "required": ["query"]
        }
    },
    {
        "name": "run_sql",
        "description": "Execute a SQL query. Only call this AFTER validate_sql confirms the query is VALID.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"}
            },
            "required": ["query"]
        }
    }
]

def run_agent(question: str, tools: list, system: str = None) -> str:
    messages = [{"role": "user", "content": question}]
    system_prompt = system or "You are a SQL assistant. Follow tool instructions carefully."
    call_log = []

    for _ in range(15):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0,
            system=system_prompt,
            tools=tools,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            return (
                next((b.text for b in response.content if b.type == "text"), ""),
                call_log
            )

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    call_log.append(block.name)
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

    return ("Max iterations reached", call_log)

print("=== EXERCISE 4A: Tool call sequence ===")
answer, call_log = run_agent(
    "How many orders have status 'complete'?",
    TOOLS_V1
)
print(f"Tool call order: {' → '.join(call_log)}")
print(f"Answer: {answer}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4B — tool_choice: force a specific tool
# Goal: see how tool_choice overrides model's own selection
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 4B: Forced tool_choice ===")

# Force the model to call get_schema on the very first call
response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=512,
    tools=TOOLS_V1,
    tool_choice={"type": "tool", "name": "get_schema"},  # forced
    messages=[{"role": "user", "content": "Help me understand the data."}]
)
print(f"stop_reason: {response.stop_reason}")
for block in response.content:
    if block.type == "tool_use":
        print(f"Tool called: {block.name} (forced by tool_choice)")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4C — Bad description vs good description
# Goal: see how vague descriptions lead to wrong tool selection
# ─────────────────────────────────────────────────────────────────────────────

TOOLS_BAD_DESCRIPTIONS = [
    {
        "name": "get_schema",
        "description": "Gets schema.",                      # too vague
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "run_sql",
        "description": "Runs SQL.",                         # too vague
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
]

print("=== EXERCISE 4C: Bad vs Good Descriptions ===")
_, bad_log = run_agent("How many users are there?", TOOLS_BAD_DESCRIPTIONS)
_, good_log = run_agent("How many users are there?", TOOLS_V1)

print(f"Bad descriptions  — tool order: {' → '.join(bad_log)}")
print(f"Good descriptions — tool order: {' → '.join(good_log)}")
print("Did the bad descriptions skip get_schema?", "get_schema" not in bad_log)
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 4D — Handle tool errors gracefully
# Goal: see that returning an error string lets the model recover
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 4D: Tool Error Recovery ===")

def run_sql_with_bad_table(query: str) -> str:
    """This always fails — simulating a missing table."""
    return "ERROR: table 'nonexistent_table' does not exist"

def execute_tool_error_demo(name: str, inputs: dict) -> str:
    if name == "get_schema":
        return get_schema()
    if name == "run_sql":
        return run_sql_with_bad_table(inputs["query"])  # always errors
    return f"ERROR: Unknown tool '{name}'"

def run_agent_error_demo(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    for _ in range(10):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0,
            system="You are a SQL assistant. If a query fails, explain why to the user.",
            tools=TOOLS_V1,
            messages=messages
        )
        messages.append({"role": "assistant", "content": response.content})
        if response.stop_reason == "end_turn":
            return next((b.text for b in response.content if b.type == "text"), "")
        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool_error_demo(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})
    return "Max iterations"

answer = run_agent_error_demo("How many rows in the orders table?")
print(f"Model response when tool always errors:\n{answer}")