import anthropic
from dotenv import load_dotenv

load_dotenv()

client = anthropic.Anthropic()
messages = []

turns = [
    "My name is Arun. I'm a data engineer based in Sydney.",
    "I specialise in Databricks and I've built AI agents.",
    "What do I do for a living?",
    "What city am I in?",
    "Summarise everything you know about me."
]

for i, user_msg in enumerate(turns):
    messages.append({"role": "user", "content": user_msg})
    response = client.messages.create(
        model="claude-haiku-4-5", max_tokens=512, messages=messages
    )
    assistant_text = response.content[0].text
    messages.append({"role": "assistant", "content": assistant_text})

    cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
    print(f"Turn {i+1} | Messages: {len(messages)} | In: {response.usage.input_tokens} | Out: {response.usage.output_tokens} | Cost: ${cost:.6f}")
    print(f"  User: {user_msg}")
    print(f"  Claude: {assistant_text[:120]}...\n")