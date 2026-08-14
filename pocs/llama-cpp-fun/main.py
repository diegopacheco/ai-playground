import json
import os
import sys
from typing import TypedDict, cast
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


MODEL = "ggml-org/Qwen3.6-27B-GGUF:Q8_0"


class Message(TypedDict):
    role: str
    content: str


class ChatRequest(TypedDict):
    model: str
    messages: list[Message]
    max_tokens: int


class Choice(TypedDict):
    message: Message


class ChatResponse(TypedDict):
    choices: list[Choice]


def chat(prompt: str) -> str:
    base_url = os.getenv("LLAMA_CPP_BASE_URL", "http://127.0.0.1:8080/v1")
    payload: ChatRequest = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
    }
    request = Request(
        f"{base_url}/chat/completions",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request) as response:
        result = cast(ChatResponse, json.load(response))
    return result["choices"][0]["message"]["content"]


def main() -> None:
    prompt = " ".join(sys.argv[1:]).strip() or input("Prompt: ").strip()
    if not prompt:
        raise ValueError("Prompt cannot be empty")
    try:
        print(chat(prompt))
    except HTTPError as error:
        details = error.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"llama.cpp returned HTTP {error.code}: {details}") from error
    except URLError as error:
        raise RuntimeError(f"Could not reach llama.cpp: {error.reason}") from error


if __name__ == "__main__":
    main()
