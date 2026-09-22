import json
import os
import sys
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-sdk"))

from agent_sdk import AgyAgent, ClaudeCodeAgent, CodexAgent, OllamaAgent

AGENTS = {"claude": ClaudeCodeAgent, "codex": CodexAgent, "agy": AgyAgent, "ollama": OllamaAgent}

MODELS = {
    "claude": ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"],
    "codex": ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-luna"],
    "agy": ["gemini-3.8-flash", "gemini-3.1-pro"],
    "ollama": ["llama3.2", "tinyllama"],
}


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def message_turn(message):
    role = message.get("role", "user")
    text = message_text(message.get("content"))
    calls = [call.get("function", {}) for call in message.get("tool_calls") or []]
    if calls:
        text = "\n".join([text] + [f"called tool {call.get('name')} with {call.get('arguments')}" for call in calls]).strip()
    if role == "tool":
        return "tool result", text
    return role, text


def build_prompt(messages):
    turns = [message_turn(message) for message in messages]
    turns = [(role, text) for role, text in turns if text.strip()]
    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]
    return "\n\n".join(f"{role}: {text}" for role, text in turns)


def tool_specs(tools):
    specs = [tool.get("function", {}) for tool in tools if isinstance(tool, dict) and tool.get("type") == "function"]
    return [{"name": spec.get("name"), "description": spec.get("description", ""), "parameters": spec.get("parameters", {})} for spec in specs if spec.get("name")]


def tool_prompt(specs, conversation):
    return "\n\n".join([
        "You have these tools:",
        json.dumps(specs),
        "Rules: use the tools for every arithmetic step, never do the math yourself. "
        "To call tools, reply with only this JSON and nothing else: "
        '{"tool_calls": [{"name": "<tool name>", "arguments": {...}}]}. '
        "Call only tools whose inputs are already known, wait for their results before the next step. "
        "When the tool results already answer the question, reply with the final answer in plain text, no JSON.",
        "Conversation:",
        conversation,
    ])


def parse_tool_calls(text, names):
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        return []
    try:
        value = json.loads(text[start : end + 1])
    except json.JSONDecodeError:
        return []
    calls = value.get("tool_calls") if isinstance(value, dict) else None
    if not isinstance(calls, list):
        return []
    valid = [call for call in calls if isinstance(call, dict) and call.get("name") in names and isinstance(call.get("arguments", {}), dict)]
    return [{"id": f"call_{uuid.uuid4().hex[:12]}", "type": "function", "function": {"name": call["name"], "arguments": json.dumps(call.get("arguments", {}))}} for call in valid]


def completion(model, text, tool_calls=None):
    message = {"role": "assistant", "content": text}
    if tool_calls:
        message = {"role": "assistant", "content": None, "tool_calls": tool_calls}
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "tool_calls" if tool_calls else "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def model_list(provider):
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": provider} for model in MODELS[provider]]}


def error(status, message):
    return status, {"error": {"message": message, "type": "bridge_error", "code": status}}


def route(method, path, body, agents):
    parts = [part for part in path.split("?")[0].split("/") if part]
    if method == "GET" and parts == ["health"]:
        return 200, {"status": "ok", "providers": sorted(agents)}
    if len(parts) < 3 or parts[1] != "v1" or parts[0] not in agents:
        return error(404, f"unknown path {path}")
    provider = parts[0]
    if method == "GET" and parts[2:] == ["models"]:
        return 200, model_list(provider)
    if method != "POST" or parts[2:] != ["chat", "completions"]:
        return error(404, f"unknown path {path}")
    if body.get("stream"):
        return error(400, "streaming is not supported by CLI agents")
    model = str(body.get("model", "")).split("/")[-1]
    prompt = build_prompt(body.get("messages", []))
    specs = tool_specs(body.get("tools") or [])
    if specs:
        prompt = tool_prompt(specs, prompt)
    try:
        text = agents[provider].call(model, prompt)
    except ValueError as failure:
        return error(400, str(failure))
    except Exception as failure:
        return error(502, str(failure))
    return 200, completion(model, text.strip(), parse_tool_calls(text, {spec["name"] for spec in specs}))


def handler(agents):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(*route("GET", self.path, {}, agents))

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.respond(*error(400, "body must be JSON"))
                return
            self.respond(*route("POST", self.path, body, agents))

        def respond(self, status, payload):
            data = json.dumps(payload).encode("utf-8")
            try:
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)
            except (BrokenPipeError, ConnectionResetError):
                sys.stderr.write(f"bridge client closed before response {self.path}\n")

        def log_message(self, fmt, *args):
            sys.stderr.write(f"bridge {self.address_string()} {fmt % args}\n")

    return Handler


def main():
    port = int(os.environ.get("BRIDGE_PORT", "8091"))
    agents = {name: agent() for name, agent in AGENTS.items()}
    server = ThreadingHTTPServer(("127.0.0.1", port), handler(agents))
    print(f"bridge listening on http://127.0.0.1:{port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
