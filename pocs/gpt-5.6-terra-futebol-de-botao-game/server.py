import json
import os
import re
import shutil
import subprocess
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


COMMANDS = {
    "claude": ["claude", "-p"],
    "codex": ["codex", "exec", "--ephemeral", "--skip-git-repo-check", "--sandbox", "read-only"],
    "agy": ["agy", "--print"],
}


def number(value, minimum, maximum):
    return max(minimum, min(maximum, float(value)))


def game_prompt(data):
    players = data.get("players", [])
    ball = data.get("ball", {})
    safe_players = [
        {"number": int(player["number"]), "x": number(player["x"], 0, 1200), "y": number(player["y"], 0, 700)}
        for player in players[:10]
    ]
    safe_ball = {"x": number(ball.get("x", 600), 0, 1200), "y": number(ball.get("y", 350), 0, 700)}
    state = json.dumps({"players": safe_players, "ball": safe_ball}, separators=(",", ":"))
    return f"""You control Inter, the red team, in a button football game. Inter attacks the left goal at x=0. Choose one red button and a shot direction. The field is 1200 by 700. Angle 0 points right, 3.1416 points left, -1.5708 points up. Power is from 0.25 to 1. Return only valid JSON with this exact shape: {{\"player\":1,\"angle\":3.1416,\"power\":0.8}}. Do not use tools. Do not explain. Current state: {state}"""


def commentary_prompt(data):
    language = "Brazilian Portuguese" if data.get("language") == "pt-BR" else "English"
    event = "a goal" if data.get("event") == "goal" else "a completed move"
    team = data.get("team") if data.get("team") in {"Grêmio", "Inter"} else "the team"
    names = {"Danrlei", "Arce", "Dinho", "C. Miguel", "P. Nunes", "André", "Gamarra", "Enciso", "Arílson", "Fabiano", "Grêmio", "Inter"}
    player = data.get("player") if data.get("player") in names else "the player"
    start = data.get("startBall", {})
    end = data.get("endBall", {})
    start_x = number(start.get("x", 600), 0, 1200)
    start_y = number(start.get("y", 350), 0, 700)
    end_x = number(end.get("x", 600), 0, 1200)
    end_y = number(end.get("y", 350), 0, 700)
    return f"""You are a lively radio narrator for a Brazilian button football match. Narrate one event in {language}. Make it vivid, colorful, playful, and genuinely fun. Use one energetic sentence of at most 30 words. Mention {player} and {team}. No insults, markdown, labels, quotes, or explanation. The event is {event}. The ball moved from ({start_x:.0f},{start_y:.0f}) to ({end_x:.0f},{end_y:.0f}) on a 1200 by 700 field. Return only the spoken commentary. Do not use tools."""


def clean_commentary(output):
    text = re.sub(r"\x1b\[[0-9;]*m", "", output)
    text = re.sub(r"\s+", " ", text).strip().strip("`\"'")
    if not text:
        raise ValueError("The narrator returned no commentary")
    return text[:500]


def parse_shot(output):
    candidates = [output.strip(), *re.findall(r"\{[^{}]+\}", output)]
    for candidate in reversed(candidates):
        try:
            shot = json.loads(candidate)
            return {
                "player": int(number(shot["player"], 1, 5)),
                "angle": number(shot["angle"], -6.2832, 6.2832),
                "power": number(shot["power"], .25, 1),
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError):
            continue
    raise ValueError("The agent did not return a valid shot")


class GameHandler(SimpleHTTPRequestHandler):
    def send_json(self, status, data):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        if self.path not in {"/api/ai-shot", "/api/commentary"}:
            self.send_json(404, {"error": "Not found"})
            return
        origin = self.headers.get("Origin")
        allowed_origins = {f"http://127.0.0.1:{self.server.server_port}", f"http://localhost:{self.server.server_port}"}
        if origin and origin not in allowed_origins:
            self.send_json(403, {"error": "Origin is not allowed"})
            return
        if not self.headers.get("Content-Type", "").startswith("application/json"):
            self.send_json(415, {"error": "JSON is required"})
            return
        try:
            length = min(int(self.headers.get("Content-Length", "0")), 65536)
            data = json.loads(self.rfile.read(length))
            provider = data.get("provider")
            command = COMMANDS.get(provider)
            if not command or not shutil.which(command[0]):
                self.send_json(503, {"error": f"{provider} CLI is not available"})
                return
            prompt = game_prompt(data) if self.path == "/api/ai-shot" else commentary_prompt(data)
            result = subprocess.run(
                [*command, prompt],
                cwd="/tmp",
                capture_output=True,
                text=True,
                timeout=60,
                env=os.environ,
            )
            if result.returncode != 0:
                message = result.stderr.strip().splitlines()[-1] if result.stderr.strip() else "Agent command failed"
                self.send_json(502, {"error": message})
                return
            if self.path == "/api/ai-shot":
                self.send_json(200, parse_shot(result.stdout))
            else:
                self.send_json(200, {"commentary": clean_commentary(result.stdout)})
        except subprocess.TimeoutExpired:
            self.send_json(504, {"error": "Agent response timed out"})
        except (KeyError, ValueError, TypeError, json.JSONDecodeError) as error:
            self.send_json(400, {"error": str(error)})

    def do_GET(self):
        if self.path == "/health":
            self.send_json(200, {"status": "ok", "game": "mesa12"})
            return
        super().do_GET()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8091
    ThreadingHTTPServer(("127.0.0.1", port), GameHandler).serve_forever()
