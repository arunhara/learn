import anthropic
from dotenv import load_dotenv


load_dotenv()

client = anthropic.Anthropic()
response = client.messages.create(
    model="claude-haiku-4-5",
    max_tokens=20,
    messages=[{"role": "user", "content": "Explain the complete medallion architecture in Databricks with Bronze, Silver, and Gold layers."}]
)

print(f"stop_reason:    {response.stop_reason}")
print(f"output_tokens:  {response.usage.output_tokens}")
print(f"response text:  '{response.content[0].text}'")