from ollama import chat

response = chat(
    model='tinyllama',
    messages=[{'role': 'user', 'content': 'How much is 1 + 1?'}],
)
print(response.message.content)