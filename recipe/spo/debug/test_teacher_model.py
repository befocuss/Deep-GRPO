from openai import OpenAI


client = OpenAI(
  api_key="my_key",
  base_url="http://placeholder-api-server:8000/v1"
)

messages = [{
  "role": "user",
  "content": "Hello"
}]

response = client.chat.completions.create(
                    model="Qwen3-235B-A22B-Instruct-2507-AWQ",
                    messages=messages,
                    timeout=30
                )

print(response)