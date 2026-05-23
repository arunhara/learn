import os
import time
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()
client = Anthropic()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5A — Context window growth simulation
# Goal: show how a long agent loop pushes toward context limits
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 5A: Context Growth Simulation ===")

def simulate_long_loop(n_turns: int):
    """Simulate a multi-turn conversation and track token growth."""
    messages = [{"role": "user", "content": "Start a long conversation."}]
    token_history = []

    for i in range(n_turns):
        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=100,
            messages=messages + [{"role": "user", "content": f"Turn {i+1}: say something brief."}]
        )
        token_history.append(response.usage.input_tokens)
        messages.append({"role": "user", "content": f"Turn {i+1}: say something brief."})
        messages.append({"role": "assistant", "content": response.content[0].text})

    print(f"{'Turn':<6} {'Input Tokens':<15} {'% of 200k limit'}")
    print("-" * 40)
    for i, tokens in enumerate(token_history):
        pct = tokens / 200_000 * 100
        bar = "█" * int(pct / 2)
        print(f"{i+1:<6} {tokens:<15} {pct:.2f}% {bar}")

simulate_long_loop(5)
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5B — No persistence: state dies with the process
# Goal: demonstrate why you need a checkpointer (Day 2 topic)
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 5B: No Persistence ===")

# Simulate a multi-turn agent session
session_messages = [
    {"role": "user", "content": "My name is Alice and I am analyzing Q1 sales data."}
]
r = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=128,
    messages=session_messages
)
session_messages.append({"role": "assistant", "content": r.content[0].text})

print(f"Session established. Messages in memory: {len(session_messages)}")
print("Simulating process restart...")

# Process restart = messages array is gone
session_messages = []

session_messages.append({"role": "user", "content": "What data was I analyzing?"})
r2 = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=128,
    messages=session_messages
)
print(f"After restart, model says: {r2.content[0].text}")
print("→ The model has no memory. History was in RAM only.")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 5C — Manual retry logic complexity
# Goal: show how retry logic inside a loop gets messy fast
#       (contrast with LangGraph's clean conditional edge approach)
# ─────────────────────────────────────────────────────────────────────────────

print("=== EXERCISE 5C: Manual Retry Logic ===")

# Simulated SQL executor that fails the first 2 times
attempt_counter = {"n": 0}

def flaky_sql(query: str) -> str:
    attempt_counter["n"] += 1
    if attempt_counter["n"] <= 2:
        return f"ERROR: connection timeout (attempt {attempt_counter['n']})"
    return "id | total\n1  | 249.49\n2  | 25.00"

SIMPLE_TOOLS = [
    {
        "name": "run_sql",
        "description": "Execute SQL. On error, the model should retry with the same or corrected query.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"]
        }
    }
]

# Manual retry state — this is what you manage yourself in the raw loop
retry_count = 0
MAX_RETRIES = 3
messages = [{"role": "user", "content": "Show me total orders per user."}]

for iteration in range(15):
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=512,
        temperature=0,
        system=f"You are a SQL assistant. Schema: orders(id, user_id, total). You have {MAX_RETRIES - retry_count} retries remaining.",
        tools=SIMPLE_TOOLS,
        messages=messages
    )
    messages.append({"role": "assistant", "content": response.content})

    if response.stop_reason == "end_turn":
        final = next((b.text for b in response.content if b.type == "text"), "")
        print(f"Completed after {iteration+1} iterations, {retry_count} retries")
        print(f"Answer: {final}")
        break

    if response.stop_reason == "tool_use":
        tool_results = []
        for block in response.content:
            if block.type == "tool_use":
                result = flaky_sql(block.input["query"])
                print(f"  Tool call attempt {attempt_counter['n']}: {result[:60]}")

                if result.startswith("ERROR"):
                    retry_count += 1          # manual state management
                    if retry_count >= MAX_RETRIES:
                        print(f"  Max retries reached. Escalating.")
                        print(f"  → In LangGraph this would be a conditional edge to error_handler")
                        break

                tool_results.append({
                    "type": "tool_result",
                    "tool_use_id": block.id,
                    "content": result
                })
        else:
            messages.append({"role": "user", "content": tool_results})
            continue
        break

print()
print("Notice:")
print("  - retry_count is a raw variable we manage manually")
print("  - escalation is a 'break' buried inside nested loops")
print("  - there is no way to 'resume from the retry point' after a crash")
print("  - In LangGraph: retry_count is a state field, escalation is a named edge")