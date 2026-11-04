import argparse
import random
import signal
import subprocess
import sys
import time
from collections import deque
from collections.abc import Callable
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "mcp"))

from mcpkit.client import McpError, StdioClient

DECISION = ROOT / "skills" / "decision-log" / "scripts" / "decision.py"
GRADER = ROOT / "skills" / "answer-grader" / "scripts" / "grade.py"
FIXTURES = ROOT / "skills" / "answer-grader" / "evals" / "fixtures"
TOPICS = ["cache", "queue", "retries", "tracing", "sharding", "rate-limits", "schemas", "timeouts"]
TAGS = ["infra", "docs", "mcp", "loki", "postgres", "oauth", "evals"]


class Traffic:
    def __init__(self, keep: int) -> None:
        self.keep = keep
        self.bookmarks = StdioClient(ROOT / "mcp" / "bookmarks-mcp" / "run.sh", "bookmarks-mcp").open()
        self.flags = StdioClient(ROOT / "mcp" / "feature-flags-mcp" / "run.sh", "feature-flags-mcp").open()
        self.saved_urls: deque[str] = deque()
        self.decisions: deque[str] = deque()
        self.counter = 0
        self.flags.call("set_flag", {"name": "traffic-checkout", "enabled": True, "rollout_percent": 50})
        print(f"mcp sessions open: bookmarks-mcp {self.bookmarks.tools}, feature-flags-mcp {self.flags.tools}", flush=True)

    def _skill(self, script: Path, *args: str) -> bool:
        return subprocess.run([sys.executable, str(script), *args], capture_output=True, text=True, timeout=60).returncode == 0

    def save_bookmark(self) -> tuple[str, bool]:
        url = f"https://traffic.local/{random.choice(TOPICS)}/{self.counter}"
        ok, _ = self.bookmarks.call("save_bookmark", {"url": url, "title": f"Notes on {url.split('/')[3]}", "tags": random.sample(TAGS, 2)})
        self.saved_urls.append(url)
        while len(self.saved_urls) > self.keep:
            self.bookmarks.call("delete_bookmark", {"url": self.saved_urls.popleft()})
        return "mcp bookmarks-mcp save_bookmark", ok

    def search_bookmarks(self) -> tuple[str, bool]:
        ok, _ = self.bookmarks.call("search_bookmarks", {"query": random.choice(TOPICS)})
        return "mcp bookmarks-mcp search_bookmarks", ok

    def set_flag(self) -> tuple[str, bool]:
        ok, _ = self.flags.call("set_flag", {"name": "traffic-checkout", "enabled": random.random() > 0.15, "rollout_percent": random.randint(0, 100)})
        return "mcp feature-flags-mcp set_flag", ok

    def evaluate_flag(self) -> tuple[str, bool]:
        ok, _ = self.flags.call("evaluate_flag", {"name": "traffic-checkout", "user_id": f"user-{random.randint(1, 500)}"})
        return "mcp feature-flags-mcp evaluate_flag", ok

    def add_decision(self) -> tuple[str, bool]:
        title = f"Traffic {random.choice(TOPICS)} {self.counter}"
        ok = self._skill(DECISION, "add", "--title", title, "--decision", "Adopt it", "--why", "generated traffic")
        self.decisions.append(title.lower().replace(" ", "-"))
        while len(self.decisions) > self.keep:
            self._skill(DECISION, "remove", self.decisions.popleft())
        return "skill decision-log add", ok

    def grade_answer(self) -> tuple[str, bool]:
        answer = FIXTURES / ("bad.md" if random.random() < 0.25 else "good.md")
        ok = self._skill(GRADER, "grade", "--suite", "traffic-release-notes", "--answer", str(answer), "--checks", str(FIXTURES / "checks.json"))
        return f"skill answer-grader grade {answer.name}", ok

    def step(self) -> tuple[str, bool]:
        self.counter += 1
        actions: list[Callable[[], tuple[str, bool]]] = [
            self.save_bookmark,
            self.search_bookmarks,
            self.set_flag,
            self.evaluate_flag,
            self.add_decision,
            self.grade_answer,
        ]
        return random.choice(actions)()

    def cleanup(self) -> None:
        for url in self.saved_urls:
            self.bookmarks.call("delete_bookmark", {"url": url})
        for decision in self.decisions:
            self._skill(DECISION, "remove", decision)
        self.flags.call("delete_flag", {"name": "traffic-checkout"})
        self.bookmarks.close()
        self.flags.close()


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate real traffic through the MCP servers and the skills, no Claude involved")
    parser.add_argument("--rounds", type=int, default=0, help="stop after N actions, 0 runs until Ctrl+C")
    parser.add_argument("--interval", type=float, default=0.7, help="seconds between actions, at most 1")
    parser.add_argument("--keep", type=int, default=10, help="bookmarks and decisions kept alive before the oldest is deleted")
    args = parser.parse_args()
    interval = min(max(args.interval, 0.1), 1.0)
    stop = {"now": False}
    signal.signal(signal.SIGINT, lambda *_: stop.update(now=True))
    signal.signal(signal.SIGTERM, lambda *_: stop.update(now=True))
    traffic = Traffic(args.keep)
    done = 0
    try:
        while not stop["now"] and (args.rounds == 0 or done < args.rounds):
            started = time.monotonic()
            try:
                label, ok = traffic.step()
            except (McpError, subprocess.TimeoutExpired) as error:
                label, ok = f"error {error}", False
            done += 1
            print(f"{done:>6} {'ok  ' if ok else 'FAIL'} {int((time.monotonic() - started) * 1000):>5}ms {label}", flush=True)
            time.sleep(interval)
    finally:
        print("cleaning up traffic data", flush=True)
        traffic.cleanup()
    print(f"traffic stopped after {done} actions", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
