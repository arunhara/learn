from anthropic import Anthropic
from dotenv import load_dotenv

load_dotenv()  # reads .env and loads ANTHROPIC_API_KEY into os.environ

client = Anthropic()


response = client.messages.create(
    model="claude-haiku-4-5-20251001",
    max_tokens=256,
    system="You are a data architecture consultant.",
    messages=[{"role": "user", "content": "What is a medallion architecture in one sentence?"}]
)

print(f"id:             {response.id}")
print(f"type:           {response.type}")
print(f"role:           {response.role}")
print(f"model:          {response.model}")
print(f"stop_reason:    {response.stop_reason}")
print(f"stop_sequence:  {response.stop_sequence}")
print(f"content blocks: {len(response.content)}")
for i, block in enumerate(response.content):
    print(f"  block {i}: type={block.type}")
    if block.type == "text":
        print(f"           text={block.text[:100]}...")
print(f"input_tokens:   {response.usage.input_tokens}")
print(f"output_tokens:  {response.usage.output_tokens}")
print(f"cache_creation: {response.usage.cache_creation_input_tokens}")
print(f"cache_read:     {response.usage.cache_read_input_tokens}")

# Calculate cost
cost = (response.usage.input_tokens * 3 + response.usage.output_tokens * 15) / 1_000_000
print(f"cost:           ${cost:.6f}")