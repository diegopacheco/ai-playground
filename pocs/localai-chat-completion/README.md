### Run LocalAI
```
podman run --rm -p 8080:8080 --name local-ai -ti localai/localai:latest-aio-cpu
```

### Result
Input:
```
 "messages": [{"role": "user", "content": "Say this is a test!"}],
 "temperature": 0.7,
 "prompt": "What is this?",
```
Output:
```
{
  "created": 1714300026,
  "object": "chat.completion",
  "id": "8e46a587-b6b0-432d-9921-dda302450429",
  "choices": [
    {
      "index": 0,
      "finish_reason": "stop",
      "message": {
        "role": "assistant",
        "content": "\n<dummy00022>, you're right that it's a test!\n\n<dummy00029>?\n\n### Instructions\n\n-?\n-?\n-?\n\nIf you say \"this is a test,\" you're right. But I'm not sure what you're testing for. Can you clarify what you're referring to?\n\n-?\n-?\n-?\n\nI'm not sure what you're referring to. Can you please provide more context?\n\n-?\n-?\n-?\n\nIt's possible that I misunderstood something. Can you please explain in more detail?\n\n-?\n-?\n-?\n\nCertainly! I was referring to the idea that the current situation is being treated as a test, possibly to see how people will react or what decisions they will make. However, without more context, it's difficult to determine what specific elements of the situation are being tested.\n\n-?\n-?\n-?\n\nIf you can provide more information about the context, that would be helpful. What exactly is being tested and what is the purpose of the test?<|im_end|>"
      }
    }
  ],
  "usage": {
    "prompt_tokens": 7,
    "completion_tokens": 245,
    "total_tokens": 252
  }
}
```
