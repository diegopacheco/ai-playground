import requests
import json

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "messages": [{"role": "user", "content": "Say this is a test!"}],
    "temperature": 0.7,
    "prompt": "What is this?",
}

response = requests.post(url, headers=headers, data=json.dumps(data))

# print the response pretty and formated
print(json.dumps(response.json(), indent=2))