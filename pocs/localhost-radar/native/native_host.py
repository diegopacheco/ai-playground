#!/usr/bin/env python3
import json
import fcntl
import os
import re
import signal
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager

PROTECTED_NAMES = {"Google Chrome", "WindowServer", "chrome", "gvproxy", "kernel_task", "launchd", "loginwindow", "podman"}
TOOL_PATHS = ["/opt/homebrew/bin", "/usr/local/bin", "/opt/local/bin", "/usr/bin", "/bin", "/usr/sbin", "/sbin"]


@contextmanager
def podman_guard():
    lock_path = os.path.join(tempfile.gettempdir(), "localhost-radar-podman.lock")
    with open(lock_path, "a", encoding="utf-8") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


def executable(name):
    search_path = os.pathsep.join(TOOL_PATHS + [os.environ.get("PATH", "")])
    return shutil.which(name, path=search_path) or name


def run(command, timeout=10):
    resolved = [executable(command[0]), *command[1:]]
    try:
        result = subprocess.run(resolved, capture_output=True, text=True, timeout=timeout, check=False)
        return result.returncode, result.stdout, result.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"{command[0]} is not installed or not available in PATH."
    except PermissionError:
        return 126, "", f"{command[0]} is not permitted in this environment."
    except OSError as exception:
        return 1, "", str(exception)
    except subprocess.TimeoutExpired:
        return 124, "", f"{command[0]} did not respond within {timeout} seconds."


def normalize_names(value):
    if isinstance(value, list):
        return value[0] if value else "unnamed"
    return str(value or "unnamed")


def normalize_ports(value):
    if not value:
        return ""
    if isinstance(value, str):
        return value
    ports = []
    for item in value if isinstance(value, list) else [value]:
        if not isinstance(item, dict):
            ports.append(str(item))
            continue
        host = item.get("host_port") or item.get("HostPort") or ""
        container = item.get("container_port") or item.get("ContainerPort") or ""
        protocol = item.get("protocol") or item.get("Protocol") or "tcp"
        if host and container:
            ports.append(f"{host}->{container}/{protocol}")
        elif container:
            ports.append(f"{container}/{protocol}")
    return ", ".join(ports)


def list_containers():
    code, output, error = run(["podman", "ps", "--all", "--format", "json"])
    if code != 0 and podman_connection_error(error):
        start_podman_machine()
        code, output, error = wait_for_podman()
    if code != 0:
        raise RuntimeError(error or "Podman could not list containers.")
    try:
        raw = json.loads(output or "[]")
    except json.JSONDecodeError as exception:
        raise RuntimeError("Podman returned invalid container data.") from exception
    containers = []
    for item in raw:
        state = str(item.get("State") or item.get("state") or "unknown").lower()
        containers.append({
            "id": str(item.get("Id") or item.get("ID") or item.get("id") or ""),
            "image": str(item.get("Image") or item.get("image") or "unknown"),
            "name": normalize_names(item.get("Names") or item.get("names")),
            "ports": normalize_ports(item.get("Ports") or item.get("ports")),
            "running": state == "running",
            "state": state,
            "status": str(item.get("Status") or item.get("status") or state)
        })
    return sorted(containers, key=lambda item: (not item["running"], item["name"].lower()))


def podman_machine_state():
    code, output, _ = run(["podman", "machine", "inspect"])
    if code != 0:
        return "unknown"
    try:
        machines = json.loads(output or "[]")
        return str(machines[0].get("State") or "unknown").lower() if machines else "unknown"
    except json.JSONDecodeError:
        return "unknown"


def podman_connection_error(error):
    value = str(error).lower()
    return any(phrase in value for phrase in ["cannot connect to podman", "connection refused", "connection reset by peer", "handshake failed"])


def wait_for_podman():
    last = (1, "", "Podman machine did not become ready.")
    for _ in range(15):
        last = run(["podman", "ps", "--all", "--format", "json"])
        if last[0] == 0:
            return last
        time.sleep(1)
    return last


def start_podman_machine():
    code, _, _ = run(["podman", "ps", "--all", "--format", "json"])
    if code == 0:
        return {"state": "running"}
    state = podman_machine_state()
    if state == "running":
        stop_code, _, stop_error = run(["podman", "machine", "stop"], 30)
        if stop_code != 0:
            raise RuntimeError(stop_error or "The stale Podman machine could not stop.")
    code, output, error = run(["podman", "machine", "start"], 40)
    if code != 0:
        raise RuntimeError(error or "Podman machine could not start.")
    return {"state": "running", "output": output.strip()}


def process_path(pid):
    code, output, _ = run(["lsof", "-a", "-p", str(pid), "-d", "cwd", "-Fn"], 3)
    if code not in (0, 1):
        return ""
    paths = [line[1:] for line in output.splitlines() if line.startswith("n")]
    return paths[0] if paths else ""


def process_command(pid):
    code, output, _ = run(["ps", "-p", str(pid), "-o", "command="], 3)
    return output.strip() if code == 0 else ""


def language_for(name, command, path):
    value = f"{name} {command} {path}".lower()
    rules = [
        (("node", "npm", "npx", "vite", "next"), "JavaScript / TypeScript"),
        (("bun",), "Bun / TypeScript"),
        (("deno",), "Deno / TypeScript"),
        (("python", "uvicorn", "gunicorn", "flask", "django"), "Python"),
        (("java", "spring", "tomcat", "gradle"), "Java / JVM"),
        (("ruby", "rails", "puma"), "Ruby"),
        (("php",), "PHP"),
        (("dotnet", "kestrel"), ".NET"),
        (("cargo", "target/debug", "target/release"), "Rust"),
        (("go-build", "golang"), "Go")
    ]
    for needles, language in rules:
        if any(needle in value for needle in needles):
            return language
    return "Native / Other"


def parse_lsof(output):
    services = []
    current_pid = None
    current_name = "unknown"
    seen = set()
    for line in output.splitlines():
        if not line:
            continue
        marker, value = line[0], line[1:]
        if marker == "p":
            current_pid = int(value) if value.isdigit() else None
        elif marker == "c":
            current_name = value
        elif marker == "n" and current_pid:
            match = re.search(r":(\d+)(?:\s|$)", value)
            if not match:
                continue
            port = int(match.group(1))
            key = (current_pid, port)
            if key in seen:
                continue
            seen.add(key)
            services.append({"pid": current_pid, "name": current_name, "port": port})
    return services


def list_services():
    code, output, error = run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN", "-Fpcn"])
    if code not in (0, 1):
        raise RuntimeError(error or "lsof could not inspect listening ports.")
    services = parse_lsof(output)
    for service in services:
        path = process_path(service["pid"])
        command = process_command(service["pid"])
        service["path"] = path
        service["language"] = language_for(service["name"], command, path)
        service["protected"] = service["name"] in PROTECTED_NAMES or service["pid"] <= 1
    return sorted(services, key=lambda item: (item["port"], item["pid"]))


def container_action(container, operation):
    if operation not in {"stop", "restart"}:
        raise RuntimeError("Unsupported container operation.")
    if not re.fullmatch(r"[a-zA-Z0-9_.-]{1,128}", str(container)):
        raise RuntimeError("Invalid container identifier.")
    code, output, error = run(["podman", operation, str(container)], 30)
    if code != 0:
        raise RuntimeError(error or f"Podman could not {operation} the container.")
    return {"container": str(container), "operation": operation, "output": output.strip()}


def kill_process(pid):
    try:
        target = int(pid)
    except (TypeError, ValueError) as exception:
        raise RuntimeError("Invalid process identifier.") from exception
    service = next((item for item in list_services() if item["pid"] == target), None)
    if not service:
        raise RuntimeError("The process is no longer listening on a local port.")
    if service["protected"]:
        raise RuntimeError("This process is protected from termination.")
    os.kill(target, signal.SIGTERM)
    return {"pid": target, "signal": "SIGTERM"}


def dispatch(message):
    action = message.get("action")
    if action == "list_containers":
        with podman_guard():
            return list_containers()
    if action == "list_services":
        return list_services()
    if action == "container_action":
        return container_action(message.get("container"), message.get("operation"))
    if action == "kill_process":
        return kill_process(message.get("pid"))
    if action == "start_podman_machine":
        with podman_guard():
            return start_podman_machine()
    raise RuntimeError("Unsupported native host action.")


def read_message():
    raw_length = sys.stdin.buffer.read(4)
    if len(raw_length) != 4:
        return None
    length = struct.unpack("=I", raw_length)[0]
    if length > 1024 * 1024:
        raise RuntimeError("Native message is too large.")
    payload = sys.stdin.buffer.read(length)
    return json.loads(payload.decode("utf-8"))


def write_message(message):
    payload = json.dumps(message, separators=(",", ":")).encode("utf-8")
    sys.stdout.buffer.write(struct.pack("=I", len(payload)))
    sys.stdout.buffer.write(payload)
    sys.stdout.buffer.flush()


def main():
    try:
        message = read_message()
        if message is None:
            return
        write_message({"ok": True, "data": dispatch(message)})
    except Exception as exception:
        write_message({"ok": False, "error": str(exception)})


if __name__ == "__main__":
    main()
