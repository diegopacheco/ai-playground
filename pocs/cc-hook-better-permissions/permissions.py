#!/usr/bin/env python3

import json
import sys
import re
import os
from pathlib import Path

HOME = str(Path.home())

CREDENTIAL_PATTERNS = [
    os.path.join(HOME, ".ssh", "id_*"),
    os.path.join(HOME, ".ssh", "config"),
    os.path.join(HOME, ".ssh", "known_hosts"),
    os.path.join(HOME, ".ssh", "authorized_keys"),
    os.path.join(HOME, ".aws", "credentials"),
    os.path.join(HOME, ".aws", "config"),
    os.path.join(HOME, ".gnupg"),
    os.path.join(HOME, ".netrc"),
    os.path.join(HOME, ".kube", "config"),
    os.path.join(HOME, ".docker", "config.json"),
    os.path.join(HOME, ".npmrc"),
    os.path.join(HOME, ".pypirc"),
    os.path.join(HOME, ".git-credentials"),
    os.path.join(HOME, ".config", "gh", "hosts.yml"),
]

CREDENTIAL_FILENAMES = [
    ".env",
    ".env.local",
    ".env.production",
    ".env.staging",
    ".env.development",
    "credentials.json",
    "secrets.yaml",
    "secrets.yml",
    "secret.yaml",
    "secret.yml",
    "token",
    "password",
    "passwords",
    "passwords.txt",
    ".git-credentials",
]

CREDENTIAL_SUBSTRINGS = [
    "id_rsa",
    "id_ed25519",
    "id_ecdsa",
    "id_dsa",
    ".ssh/",
    ".aws/credentials",
    ".gnupg/",
    ".netrc",
    ".kube/config",
    ".docker/config.json",
    ".npmrc",
    ".pypirc",
    ".git-credentials",
    "hosts.yml",
]

RM_COMMANDS = ["rm", "rmdir", "shred", "unlink"]

NETWORK_COMMANDS = [
    "curl", "wget", "nc", "ncat", "netcat",
    "ssh", "scp", "rsync", "ftp", "sftp",
    "http.server", "php -S",
]


def normalize_path(p):
    p = p.replace("$HOME", HOME).replace("~", HOME)
    p = os.path.normpath(p)
    try:
        p = str(Path(p).resolve())
    except (OSError, ValueError):
        pass
    return p


def is_credential_path(path_str):
    normalized = normalize_path(path_str)
    for pattern in CREDENTIAL_PATTERNS:
        from fnmatch import fnmatch
        if fnmatch(normalized, pattern):
            return True
        if normalized.startswith(os.path.join(HOME, ".gnupg")):
            return True
    basename = os.path.basename(normalized)
    if basename in CREDENTIAL_FILENAMES:
        return True
    if basename.startswith(".env"):
        return True
    if "secret" in basename.lower() and not basename.endswith((".py", ".js", ".ts", ".go", ".java", ".rs")):
        return True
    if "password" in basename.lower() and not basename.endswith((".py", ".js", ".ts", ".go", ".java", ".rs")):
        return True
    return False


def command_touches_credentials(command):
    for substr in CREDENTIAL_SUBSTRINGS:
        if substr in command:
            return True
    tokens = command.split()
    for token in tokens:
        if "/" in token or token.startswith("."):
            if is_credential_path(token):
                return True
    return False


def is_rm_command(command):
    tokens = command.split()
    for i, token in enumerate(tokens):
        if token in ("|", "&&", ";", "||"):
            continue
        clean = token.split("/")[-1]
        if clean in RM_COMMANDS:
            return True
    return False


def is_network_command(command):
    for cmd in NETWORK_COMMANDS:
        pattern = r'(^|[;&|]\s*)' + re.escape(cmd) + r'(\s|$)'
        if re.search(pattern, command):
            return True
    return False


def check_bash(params):
    command = params.get("command", "")
    if command_touches_credentials(command):
        return {"decision": "block", "reason": "Blocked: command accesses credential/secret files"}
    if is_rm_command(command):
        return {"decision": "ask", "message": f"rm command detected: {command}"}
    if is_network_command(command):
        return {"decision": "ask", "message": f"Network command detected: {command}"}
    return {"decision": "allow"}


def check_file_path(params):
    file_path = params.get("file_path", "")
    if file_path and is_credential_path(file_path):
        return {"decision": "block", "reason": f"Blocked: access to credential file {file_path}"}
    return {"decision": "allow"}


def check_glob(params):
    pattern = params.get("pattern", "")
    path = params.get("path", "")
    for check in [pattern, path]:
        if check:
            for substr in CREDENTIAL_SUBSTRINGS:
                if substr in check:
                    return {"decision": "block", "reason": f"Blocked: glob pattern targets credential directory"}
    return {"decision": "allow"}


def check_grep(params):
    path = params.get("path", "")
    if path and is_credential_path(path):
        return {"decision": "block", "reason": f"Blocked: grep targets credential file {path}"}
    return {"decision": "allow"}


def main():
    raw = sys.stdin.read()
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        print(json.dumps({"decision": "allow"}))
        return

    tool = data.get("tool", "")
    params = data.get("params", {}) if isinstance(data.get("params"), dict) else {}

    if tool == "Bash":
        result = check_bash(params)
    elif tool in ("Read", "Write", "Edit"):
        result = check_file_path(params)
    elif tool == "Glob":
        result = check_glob(params)
    elif tool == "Grep":
        result = check_grep(params)
    else:
        result = {"decision": "allow"}

    print(json.dumps(result))


if __name__ == "__main__":
    main()
