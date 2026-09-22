import hashlib
import io
import json
import os
import shutil
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
import zipfile
from functools import partial
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "agent-sdk"))

from agent_sdk import AgyAgent, ClaudeCodeAgent, CodexAgent, OllamaAgent, process_runner

AGENTS = {"claude": ClaudeCodeAgent, "codex": CodexAgent, "agy": AgyAgent, "ollama": OllamaAgent}

MODELS = {
    "claude": ["claude-opus-5", "claude-sonnet-5", "claude-fable-5-1"],
    "codex": ["gpt-6-astra", "gpt-5.6-sol", "gpt-5.6-luna"],
    "agy": ["gemini-3.8-flash", "gemini-3.1-pro"],
    "ollama": ["llama3.2", "tinyllama"],
}

SKILL_DIRS = (".claude/skills", ".agents/skills")
SKILLS_ZIP = "/api/skills/serve/all/download.zip"


def http_bytes(url, timeout=30):
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return response.read()
    except urllib.error.HTTPError as failure:
        with failure:
            if failure.code == 404:
                return b""
            raise


def safe_members(archive):
    members = []
    for name in archive.namelist():
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(f"unsafe path in skills archive {name}")
        if not name.endswith("/"):
            members.append(name)
    return members


class SkillSync:
    def __init__(self, bifrost_url, workspace, fetch=http_bytes):
        self.url = bifrost_url.rstrip("/") + SKILLS_ZIP
        self.workspace = Path(workspace)
        self.fetch = fetch
        self.lock = threading.Lock()
        self.digest = None

    def sync(self):
        data = self.fetch(self.url)
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            if digest != self.digest:
                self.install(data)
                self.digest = digest
        return self.names()

    def install(self, data):
        self.workspace.mkdir(parents=True, exist_ok=True)
        members = {}
        if data:
            with zipfile.ZipFile(io.BytesIO(data)) as archive:
                members = {name: archive.read(name) for name in safe_members(archive)}
        for skill_dir in SKILL_DIRS:
            target = self.workspace / skill_dir
            shutil.rmtree(target, ignore_errors=True)
            target.mkdir(parents=True)
            for name, content in members.items():
                path = target / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(content)

    def root(self):
        return self.workspace / SKILL_DIRS[-1]

    def names(self):
        return sorted(path.parent.name for path in self.root().glob("*/SKILL.md"))

    def prompt(self):
        parts = []
        for name in self.names():
            folder = self.root() / name
            parts.append(f"=== skill {name} ===\n{(folder / 'SKILL.md').read_text(encoding='utf-8')}")
            for path in sorted(folder.rglob("*")):
                if path.is_file() and path.name != "SKILL.md":
                    parts.append(f"--- file {name}/{path.relative_to(folder).as_posix()} ---\n{path.read_text(encoding='utf-8')}")
        if not parts:
            return ""
        return "\n\n".join(["You have these skills. When the question matches a skill description, follow that skill exactly.", *parts])


def message_text(content):
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") == "text")
    return ""


def build_prompt(messages):
    turns = [(message.get("role", "user"), message_text(message.get("content"))) for message in messages]
    turns = [(role, text) for role, text in turns if text.strip()]
    if len(turns) == 1 and turns[0][0] == "user":
        return turns[0][1]
    return "\n\n".join(f"{role}: {text}" for role, text in turns)


def cli_request(provider, prompt, skills):
    if provider == "ollama":
        catalog = skills.prompt()
        return (f"{catalog}\n\nQuestion:\n{prompt}" if catalog else prompt), []
    if provider == "agy":
        return prompt, ["--add-dir", str(skills.workspace)]
    return prompt, []


def completion(model, text):
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }


def model_list(provider):
    return {"object": "list", "data": [{"id": model, "object": "model", "owned_by": provider} for model in MODELS[provider]]}


def error(status, message):
    return status, {"error": {"message": message, "type": "bridge_error", "code": status}}


def route(method, path, body, agents, skills):
    parts = [part for part in path.split("?")[0].split("/") if part]
    if method == "GET" and parts == ["health"]:
        return 200, {"status": "ok", "providers": sorted(agents), "skills": skills.names()}
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
    if not prompt.strip():
        return error(400, "prompt is required")
    try:
        skills.sync()
    except (OSError, ValueError, zipfile.BadZipFile) as failure:
        return error(502, f"could not load skills from bifrost: {failure}")
    prompt, args = cli_request(provider, prompt, skills)
    try:
        text = agents[provider].call(model, prompt, args)
    except ValueError as failure:
        return error(400, str(failure))
    except Exception as failure:
        return error(502, str(failure))
    return 200, completion(model, text.strip())


class Server(ThreadingHTTPServer):
    request_queue_size = 64


def handler(agents, skills):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.respond(*route("GET", self.path, {}, agents, skills))

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self.respond(*error(400, "body must be JSON"))
                return
            self.respond(*route("POST", self.path, body, agents, skills))

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
    port = int(os.environ.get("BRIDGE_PORT", "8191"))
    workspace = Path(os.environ.get("SKILLS_WORKSPACE", Path(__file__).resolve().parent.parent / ".run" / "workspace"))
    skills = SkillSync(os.environ.get("BIFROST_URL", "http://127.0.0.1:8180"), workspace)
    runner = partial(process_runner, cwd=str(workspace))
    agents = {name: agent(runner) for name, agent in AGENTS.items()}
    workspace.mkdir(parents=True, exist_ok=True)
    server = Server(("127.0.0.1", port), handler(agents, skills))
    print(f"bridge listening on http://127.0.0.1:{port}, skills workspace {workspace}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
