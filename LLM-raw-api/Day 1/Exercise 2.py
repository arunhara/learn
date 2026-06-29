from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()

client = Anthropic()

# TEST 1: System prompt as a message (should fail)
print("TEST 1: System as message role")
try:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=100,
        messages=[
            {"role": "system", "content": "Be helpful."},
            {"role": "user", "content": "Hello"}
        ]
    )
    print(f"  Result: Succeeded (unexpected)")
except Exception as e:
    print(f"  Result: {type(e).__name__}: {str(e)[:200]}")

# TEST 2: Two consecutive user messages (should fail)
print("\nTEST 2: Consecutive user messages")
try:
    r = client.messages.create(
        model="claude-haiku-4-5-20251001", max_tokens=100,
        messages=[
            {"role": "user", "content": "Hello"},
            {"role": "user", "content": "How are you?"}
        ]
    )
    print(f"  Result: Succeeded (unexpected)")
except Exception as e:
    print(f"  Result: {type(e).__name__}: {str(e)[:200]}")

# TEST 3: Correct approach for both
print("\nTEST 3: Correct — system as parameter, single user message")
r = client.messages.create(
    model="claude-haiku-4-5-20251001", max_tokens=100,
    system="Be helpful.",
    messages=[{"role": "user", "content": "Hello. How are you?"}]
)
print(f"  Result: Success — {r.content[0].text[:80]}...")