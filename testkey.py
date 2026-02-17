from openai import OpenAI

client = OpenAI(
  base_url="https://openrouter.ai/api/v1",
  api_key="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",
)
response = client.chat.completions.create(
  extra_headers={
    "HTTP-Referer": "<YOUR_SITE_URL>",  # Optional. Site URL for rankings on openrouter.ai.
    "X-Title": "<YOUR_SITE_NAME>",     # Optional. Site title for rankings on openrouter.ai.
  },
  model="openai/gpt-5.2",
  messages=[
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"}
  ],
)
reply = response.choices[0].message
print(reply)