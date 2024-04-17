import openai

# Set the API key and base URL
openai.api_key = "lm-studio"
openai.api_base = "http://localhost:1234/v1"

completion = openai.ChatCompletion.create(
  model="TheBloke/CodeLlama-7B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Introduce yourself."}
  ],
  temperature=0.7,
)
print(completion.choices[0].message['content'])

print(openai.ChatCompletion.create(
  model="TheBloke/CodeLlama-7B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Explain to me Stable diffusion and how it works."}
  ],
  temperature=0.7,
).choices[0].message['content'])

print(openai.ChatCompletion.create(
  model="TheBloke/CodeLlama-7B-Instruct-GGUF",
  messages=[
    {"role": "system", "content": "Always answer in rhymes."},
    {"role": "user", "content": "Write a sad poem about the problems of technical debt"}
  ],
  temperature=0.7,
).choices[0].message['content'])