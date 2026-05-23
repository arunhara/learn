import os
from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads ANTHROPIC_API_KEY into os.environ

client = Anthropic()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1A — Minimal request
# Goal: see the raw response object and what it contains
# ─────────────────────────────────────────────────────────────────────────────

response = client.messages.create(
    model="claude-haiku-4-5-20251001",   # cheapest model, fine for exercises
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Say exactly: Hello from the API"}
    ]
)

print("=== EXERCISE 1A: Raw Response Object ===")
print(f"response.id:          {response.id}")
print(f"response.model:       {response.model}")
print(f"response.role:        {response.role}")
print(f"response.stop_reason: {response.stop_reason}")
print(f"response.content:     {response.content}")
print(f"response.usage:       {response.usage}")
print()

# Extract just the text
text = response.content[0].text
print(f"Extracted text: {text}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1B — The system prompt
# Goal: see that system is separate from messages, and that it controls behavior
# ─────────────────────────────────────────────────────────────────────────────

response_with_system = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a SQL expert. Every response must be a single SQL query with no explanation.",
    messages=[
        {"role": "user", "content": "Get all users created in the last 7 days"}
    ]
)

print("=== EXERCISE 1B: System Prompt Effect ===")
print(response_with_system.content[0].text)
print()

# Now run the same question WITHOUT the system prompt — see the difference
response_no_system = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    messages=[
        {"role": "user", "content": "Get all users created in the last 7 days"}
    ]
)

print("=== Without system prompt ===")
print(response_no_system.content[0].text)
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1C — temperature
# Goal: see what temperature 0 vs 1 produces for SQL generation
# Run this twice and compare outputs
# ─────────────────────────────────────────────────────────────────────────────

def generate_sql(temperature: float) -> str:
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=256,
        temperature=temperature,
        system="Return only a SQL SELECT query, nothing else.",
        messages=[
            {"role": "user", "content": "Get the top 5 users by total order amount"}
        ]
    )
    return response.content[0].text

print("=== EXERCISE 1C: Temperature Effect ===")
print(f"temperature=0 (run 1): {generate_sql(0)}")
print(f"temperature=0 (run 2): {generate_sql(0)}")   # should be identical
print(f"temperature=1 (run 1): {generate_sql(1)}")
print(f"temperature=1 (run 2): {generate_sql(1)}")   # may differ
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1D — Multi-turn conversation (messages array accumulates)
# Goal: understand that the model has no memory — the array IS the memory
# ─────────────────────────────────────────────────────────────────────────────

messages = []

# Turn 1
messages.append({"role": "user", "content": "My name is Alice and I am a data engineer."})
r1 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=128, messages=messages)
messages.append({"role": "assistant", "content": r1.content[0].text})

# Turn 2
messages.append({"role": "user", "content": "What is my name and job title?"})
r2 = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=128, messages=messages)
messages.append({"role": "assistant", "content": r2.content[0].text})

print("=== EXERCISE 1D: Multi-turn conversation ===")
print(f"Turn 2 answer: {r2.content[0].text}")
print(f"Messages array length: {len(messages)}")
print()

# Now prove that without the history, the model has no memory:
messages_fresh = [{"role": "user", "content": "What is my name and job title?"}]
r_fresh = client.messages.create(model="claude-haiku-4-5-20251001", max_tokens=128, messages=messages_fresh)
print(f"Without history: {r_fresh.content[0].text}")
print()

# ─────────────────────────────────────────────────────────────────────────────
# EXERCISE 1E — Token counting
# Goal: understand what you are paying for
# ─────────────────────────────────────────────────────────────────────────────

response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a SQL expert.",
    messages=[{"role": "user", "content": "Write a query to find duplicate emails in a users table."}]
)

print("=== EXERCISE 1E: Token Usage ===")
print(f"Input tokens:  {response.usage.input_tokens}")
print(f"Output tokens: {response.usage.output_tokens}")
print(f"Total tokens:  {response.usage.input_tokens + response.usage.output_tokens}")

# Haiku pricing (as of 2025): input $0.80/MTok, output $4.00/MTok
input_cost  = response.usage.input_tokens  * 0.80  / 1_000_000
output_cost = response.usage.output_tokens * 4.00  / 1_000_000
print(f"Approx cost:   ${input_cost + output_cost:.6f}")