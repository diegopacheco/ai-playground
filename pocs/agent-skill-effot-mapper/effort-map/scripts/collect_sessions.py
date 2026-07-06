import argparse
import json
import os
import subprocess
from datetime import datetime
from pathlib import Path


def git_root(project):
    result = subprocess.run(
        ["git", "-C", str(project), "rev-parse", "--show-toplevel"],
        capture_output=True,
        text=True,
    )
    return Path(result.stdout.strip()).resolve() if result.returncode == 0 else None


def detect_agent(value):
    if value:
        return value
    if os.environ.get("CLAUDE_CODE_ENTRYPOINT") or os.environ.get("CLAUDE_PROJECT_DIR"):
        return "claude"
    if os.environ.get("CODEX_THREAD_ID") or os.environ.get("CODEX_HOME") or os.environ.get("CODEX_CI"):
        return "codex"
    raise SystemExit("Could not detect the active agent. Pass --agent claude or --agent codex.")


def text_values(value):
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        values = []
        for item in value:
            if isinstance(item, str):
                values.append(item)
            elif isinstance(item, dict):
                for key in ("text", "output", "content"):
                    if key in item:
                        values.extend(text_values(item[key]))
                if item.get("type") == "tool_use":
                    values.append(json.dumps(item.get("input", {}), ensure_ascii=False))
        return values
    if isinstance(value, dict):
        return [json.dumps(value, ensure_ascii=False)]
    return []


def record_cwd(record):
    values = [record.get("cwd")]
    payload = record.get("payload", {})
    if isinstance(payload, dict):
        values.append(payload.get("cwd"))
    return next((value for value in values if isinstance(value, str)), None)


def belongs(cwd, project):
    if not cwd:
        return False
    try:
        path = Path(cwd).resolve()
        return path == project or project in path.parents
    except OSError:
        return False


def parse_time(value):
    if not isinstance(value, str):
        return None
    text = value.replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        return None


def claude_tokens(record):
    message = record.get("message", {})
    usage = message.get("usage") if isinstance(message, dict) else None
    if not isinstance(usage, dict):
        return 0
    keys = ("input_tokens", "output_tokens", "cache_read_input_tokens", "cache_creation_input_tokens")
    return sum(int(usage.get(key, 0) or 0) for key in keys)


def codex_tokens(record):
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return 0
    info = payload.get("info") if isinstance(payload.get("info"), dict) else payload
    usage = info.get("total_token_usage") or info.get("token_usage") or info.get("last_token_usage")
    if not isinstance(usage, dict):
        return 0
    keys = ("input_tokens", "output_tokens", "cached_input_tokens", "reasoning_output_tokens")
    return sum(int(usage.get(key, 0) or 0) for key in keys)


def claude_entry(record):
    kind = record.get("type")
    if kind not in ("user", "assistant"):
        return None
    message = record.get("message", {})
    texts = text_values(message.get("content"))
    if not texts:
        return None
    return {"role": kind, "text": "\n".join(texts), "timestamp": record.get("timestamp")}


def codex_entry(record):
    payload = record.get("payload", {})
    if not isinstance(payload, dict):
        return None
    if record.get("type") == "response_item":
        kind = payload.get("type")
        if kind == "message":
            texts = text_values(payload.get("content"))
            role = payload.get("role", "agent")
        elif kind in ("function_call", "custom_tool_call"):
            texts = text_values(payload.get("arguments") or payload.get("input"))
            role = "tool-call"
        elif kind in ("function_call_output", "custom_tool_call_output"):
            texts = text_values(payload.get("output"))
            role = "tool-result"
        else:
            return None
    elif record.get("type") == "event_msg" and payload.get("type") == "agent_message":
        texts = text_values(payload.get("message"))
        role = "assistant"
    else:
        return None
    if not texts:
        return None
    return {"role": role, "text": "\n".join(texts), "timestamp": record.get("timestamp")}


def files_for(agent):
    home = Path.home()
    root = home / (".claude/projects" if agent == "claude" else ".codex/sessions")
    return root, sorted(root.rglob("*.jsonl")) if root.exists() else []


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent", choices=("claude", "codex"))
    parser.add_argument("--project", default=os.getcwd())
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats", required=True)
    args = parser.parse_args()
    agent = detect_agent(args.agent)
    project = Path(args.project).resolve()
    repository = git_root(project)
    root, files = files_for(agent)
    sessions = 0
    entries = 0
    user_prompts = 0
    total_tokens = 0
    total_seconds = 0
    per_session = []
    output = Path(args.output)
    with output.open("w", encoding="utf-8") as target:
        for path in files:
            parsed = []
            matched = False
            tokens = 0
            times = []
            with path.open(encoding="utf-8", errors="replace") as source:
                for line in source:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if belongs(record_cwd(record), project):
                        matched = True
                    entry = claude_entry(record) if agent == "claude" else codex_entry(record)
                    if entry:
                        parsed.append(entry)
                        moment = parse_time(entry.get("timestamp"))
                        if moment:
                            times.append(moment)
                    tokens += claude_tokens(record) if agent == "claude" else codex_tokens(record)
            if not matched:
                continue
            sessions += 1
            span = int((max(times) - min(times)).total_seconds()) if len(times) >= 2 else 0
            total_tokens += tokens
            total_seconds += span
            prompts_here = 0
            for entry in parsed:
                entry["session"] = path.name
                target.write(json.dumps(entry, ensure_ascii=False) + "\n")
                entries += 1
                if entry["role"] == "user":
                    user_prompts += 1
                    prompts_here += 1
            per_session.append({"session": path.name, "tokens": tokens, "seconds": span, "prompts": prompts_here})
    stats = {
        "agent": agent,
        "project": project.name,
        "project_path": str(project),
        "git_repository": str(repository) if repository else None,
        "sessions_scanned": len(files),
        "sessions_matched": sessions,
        "entries_written": entries,
        "user_prompts": user_prompts,
        "total_tokens": total_tokens,
        "avg_tokens": round(total_tokens / sessions) if sessions else 0,
        "total_time_minutes": round(total_seconds / 60, 1),
        "avg_time_minutes": round(total_seconds / 60 / sessions, 1) if sessions else 0,
        "per_session": per_session,
        "output": str(output),
    }
    Path(args.stats).write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: stats[key] for key in stats if key != "per_session"}))


if __name__ == "__main__":
    main()
