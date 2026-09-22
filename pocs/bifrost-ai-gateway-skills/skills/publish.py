import json
import mimetypes
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

SKILLS_DIR = Path(__file__).resolve().parent
SOURCE_FILES = {"skill.json", "body.md"}


def http_json(method, url, payload=None):
    data = json.dumps(payload).encode("utf-8") if payload is not None else None
    request = urllib.request.Request(url, data=data, method=method, headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"{}")
    except urllib.error.HTTPError as failure:
        with failure:
            return failure.code, json.loads(failure.read() or b"{}")


def load_skill(folder):
    meta = json.loads((folder / "skill.json").read_text(encoding="utf-8"))
    files = [
        {"path": path.relative_to(folder).as_posix(), "source_type": "text", "content": path.read_text(encoding="utf-8"), "mime_type": mimetypes.guess_type(path.name)[0] or "text/plain"}
        for path in sorted(folder.rglob("*"))
        if path.is_file() and path.relative_to(folder).as_posix() not in SOURCE_FILES
    ]
    return {"name": folder.name, "description": meta["description"], "version": meta["version"], "skill_md_body": (folder / "body.md").read_text(encoding="utf-8"), "files": files}


def local_skills(root=SKILLS_DIR):
    return [load_skill(folder) for folder in sorted(root.iterdir()) if (folder / "skill.json").is_file()]


def publish(base_url, skills, request=http_json):
    base_url = base_url.rstrip("/")
    status, body = request("GET", f"{base_url}/api/skills?limit=100")
    if status != 200:
        raise RuntimeError(f"bifrost list skills returned {status}: {body}")
    served = {skill["name"]: skill for skill in body.get("skills", [])}
    results = []
    for skill in skills:
        current = served.get(skill["name"])
        if current is None:
            status, body = request("POST", f"{base_url}/api/skills", skill)
            action = "created"
        elif current.get("latest_version") != skill["version"]:
            update = {key: value for key, value in skill.items() if key != "name"}
            status, body = request("PUT", f"{base_url}/api/skills/{current['id']}", {**update, "serve": True})
            action = "updated"
        else:
            results.append((skill["name"], "unchanged", skill["version"]))
            continue
        if status != 200:
            raise RuntimeError(f"bifrost {action[:-1]} skill {skill['name']} returned {status}: {body}")
        results.append((skill["name"], action, skill["version"]))
    return results


def main():
    base_url = os.environ.get("BIFROST_URL", "http://127.0.0.1:8180")
    try:
        results = publish(base_url, local_skills())
    except (RuntimeError, OSError) as failure:
        sys.exit(f"ERROR: {failure}")
    for name, action, version in results:
        print(f"skill {name} {version} {action}")


if __name__ == "__main__":
    main()
