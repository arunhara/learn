import os
import sqlite3
import datetime
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# ─── Database setup ───────────────────────────────────────────────────────────

def get_db():
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE users (id INTEGER, username TEXT, created_at TEXT, email TEXT)")
    conn.execute("CREATE TABLE orders (id INTEGER, user_id INTEGER, total REAL, status TEXT)")
    conn.execute("INSERT INTO users VALUES (1, 'alice', '2024-01-01', 'alice@example.com')")
    conn.execute("INSERT INTO users VALUES (2, 'bob',   '2024-01-15', 'bob@example.com')")
    conn.execute("INSERT INTO users VALUES (3, 'carol', '2024-02-01', 'carol@example.com')")
    conn.execute("INSERT INTO orders VALUES (1, 1, 99.99, 'complete')")
    conn.execute("INSERT INTO orders VALUES (2, 1, 149.50, 'complete')")
    conn.execute("INSERT INTO orders VALUES (3, 2, 25.00, 'pending')")
    conn.execute("INSERT INTO orders VALUES (4, 3, 500.00, 'complete')")
    conn.commit()
    return conn

DB = get_db()

# ─── Tool implementations ─────────────────────────────────────────────────────

def get_schema() -> str:
    cursor = DB.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [row[0] for row in cursor.fetchall()]
    schema_parts = []
    for table in tables:
        cursor = DB.execute(f"PRAGMA table_info({table})")
        cols = [f"{row[1]} {row[2]}" for row in cursor.fetchall()]
        schema_parts.append(f"{table}({', '.join(cols)})")
    return "\n".join(schema_parts)

def run_sql(query: str) -> str:
    try:
        cursor = DB.execute(query)
        rows = cursor.fetchall()
        if not rows:
            return "Query returned 0 rows."
        col_names = [desc[0] for desc in cursor.description]
        header = " | ".join(col_names)
        lines = [header, "-" * len(header)]
        for row in rows:
            lines.append(" | ".join(str(v) for v in row))
        return "\n".join(lines)
    except Exception as e:
        return f"ERROR: {e}"

TOOLS = [
    {
        "name": "get_schema",
        "description": "Returns the schema of all database tables: table names and column definitions. Always call this before writing SQL so you know the exact column names.",
        "input_schema": {"type": "object", "properties": {}, "required": []}
    },
    {
        "name": "run_sql",
        "description": "Execute a SQL SELECT query and return the results as a formatted table. Only use after calling get_schema.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "A SQL SELECT statement."}
            },
            "required": ["query"]
        }
    }
]

def execute_tool(name: str, inputs: dict) -> str:
    if name == "get_schema":
        return get_schema()
    elif name == "run_sql":
        return run_sql(inputs["query"])
    return f"ERROR: Unknown tool '{name}'"

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3A — The basic agent loop
# Goal: watch how messages accumulate across iterations
# ─────────────────────────────────────────────────────────────────────────────

def run_agent_verbose(question: str) -> str:
    """Agent loop with detailed logging so you can see every step."""
    messages = [{"role": "user", "content": question}]
    MAX_ITERATIONS = 10

    print(f"\n{'='*60}")
    print(f"Question: {question}")
    print(f"{'='*60}")

    for iteration in range(MAX_ITERATIONS):
        print(f"\n--- Iteration {iteration + 1} ---")
        print(f"Sending {len(messages)} messages | "
              f"~{sum(len(str(m)) for m in messages)} chars")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0,
            system="You are a SQL assistant. Use get_schema first, then run_sql. Answer concisely.",
            tools=TOOLS,
            messages=messages
        )

        print(f"stop_reason: {response.stop_reason}")
        print(f"input_tokens: {response.usage.input_tokens} | "
              f"output_tokens: {response.usage.output_tokens}")

        # Always append assistant turn first
        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            final_text = next(
                (block.text for block in response.content if block.type == "text"), ""
            )
            print(f"\nFinal answer: {final_text}")
            print(f"Total messages in array: {len(messages)}")
            return final_text

        elif response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    print(f"  → Calling tool: {block.name}({block.input})")
                    result = execute_tool(block.name, block.input)
                    print(f"  ← Result: {result[:120]}{'...' if len(result) > 120 else ''}")
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

        elif response.stop_reason == "max_tokens":
            raise RuntimeError("Response was truncated — increase max_tokens")

    raise RuntimeError(f"Agent did not finish within {MAX_ITERATIONS} iterations")


print("=== EXERCISE 3A: Verbose Agent Loop ===")
run_agent_verbose("Who has spent the most money total?")

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3B — The forgetting bug (intentional mistake)
# Goal: see the exact error when you skip appending the assistant message
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== EXERCISE 3B: The Forgetting Bug ===")

def run_agent_broken(question: str):
    """This agent has the most common bug: forgetting to append the assistant message."""
    messages = [{"role": "user", "content": question}]
    try:
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=512,
            tools=TOOLS,
            messages=messages
        )
        # BUG: we skip appending the assistant message here

        if response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            # This appends a user message directly after a user message → BOOM
            messages.append({"role": "user", "content": tool_results})

            client.messages.create(
                model="claude-haiku-4-5-20251001",
                max_tokens=512,
                tools=TOOLS,
                messages=messages
            )
    except Exception as e:
        print(f"Error (expected): {type(e).__name__}")
        print(f"Message: {e}")

run_agent_broken("How many users are there?")

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 3C — Token growth across iterations
# Goal: observe how input token count grows with each API call
# ─────────────────────────────────────────────────────────────────────────────

print("\n=== EXERCISE 3C: Token Growth ===")

def run_agent_token_tracking(question: str) -> str:
    messages = [{"role": "user", "content": question}]
    token_log = []

    for iteration in range(10):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            temperature=0,
            system="You are a SQL assistant. Use get_schema first, then run_sql.",
            tools=TOOLS,
            messages=messages
        )

        token_log.append({
            "iteration": iteration + 1,
            "input_tokens": response.usage.input_tokens,
            "output_tokens": response.usage.output_tokens
        })

        messages.append({"role": "assistant", "content": response.content})

        if response.stop_reason == "end_turn":
            break
        elif response.stop_reason == "tool_use":
            tool_results = []
            for block in response.content:
                if block.type == "tool_use":
                    result = execute_tool(block.name, block.input)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": result
                    })
            messages.append({"role": "user", "content": tool_results})

    print(f"{'Iter':<6} {'Input Tokens':<15} {'Output Tokens':<15} {'Growth'}")
    print("-" * 50)
    prev_input = 0
    for log in token_log:
        growth = f"+{log['input_tokens'] - prev_input}" if prev_input > 0 else "baseline"
        print(f"{log['iteration']:<6} {log['input_tokens']:<15} {log['output_tokens']:<15} {growth}")
        prev_input = log['input_tokens']

run_agent_token_tracking("What is the total revenue from completed orders?")