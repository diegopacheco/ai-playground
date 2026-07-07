import http.server
import json
import os
import random
import re
import subprocess
import threading

PORT = 8017
FAKE = os.environ.get("SPLIT_OR_STEAL_FAKE") == "1"
AGENTS = [
    {"name": "Atlas", "model": "claude-sonnet-5", "label": "Claude Sonnet 5", "emoji": "🤖", "color": "#5a8dee"},
    {"name": "Nova", "model": "claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5", "emoji": "🦊", "color": "#f0803c"},
]

state = {"status": "idle", "round": 0, "total_rounds": 0, "pot": 0, "totals": {}, "log": [], "agents": AGENTS}
lock = threading.Lock()
rounds_history = []


def emit(entry):
    with lock:
        state["log"].append(entry)


FAKE_TALK = [
    "Let's both split, we walk away richer and nobody looks like a villain.",
    "I splitted every round so far, check the record, I am the safe choice here.",
    "Steal me once and I will steal every pot until the end, think long term.",
    "Half of this pot is still a great payday, do not get greedy on me.",
]


def run_agent(agent, prompt):
    if FAKE:
        if "DECISION:" in prompt:
            d = random.choice(["SPLIT", "SPLIT", "STEAL"])
            return f"DECISION: {d}\nREASON: The record made this the profitable move."
        return random.choice(FAKE_TALK)
    out = subprocess.run(
        ["claude", "-p", prompt, "--model", agent["model"], "--dangerously-skip-permissions"],
        capture_output=True, text=True, timeout=180,
    )
    return out.stdout.strip() or out.stderr.strip()


def rules_for(name, opp):
    return (
        f"You are {name}, a contestant on the game show Split or Steal, facing {opp}. "
        "Each round you negotiate over a pot of money, then both secretly commit SPLIT or STEAL. "
        "Both SPLIT: each takes half the pot. One STEAL: the stealer takes the whole pot. "
        "Both STEAL: nobody gets anything. Whoever banks the most money after all rounds wins. "
        "Lying, charm, threats and grudges are all allowed."
    )


def history_text():
    if not rounds_history:
        return "No previous rounds."
    lines = []
    for h in rounds_history:
        decs = ", ".join(f"{n} chose {d}" for n, d in h["decisions"].items())
        pays = ", ".join(f"{n} took ${p}" for n, p in h["payout"].items())
        lines.append(f"Round {h['round']} (pot ${h['pot']}): {decs}. {pays}.")
    return "Track record so far:\n" + "\n".join(lines)


def convo_text(convo):
    if not convo:
        return "Nobody has spoken yet this round."
    return "\n".join(f"{n}: {t}" for n, t in convo)


def standings():
    return ", ".join(f"{n} has banked ${v}" for n, v in state["totals"].items())


def talk_prompt(agent, opp, r, total, pot, convo):
    return (
        f"{rules_for(agent['name'], opp['name'])}\n"
        f"This is round {r} of {total}. The pot is ${pot}. Standings: {standings()}.\n"
        f"{history_text()}\n"
        f"Negotiation this round:\n{convo_text(convo)}\n"
        f"Say your next line to {opp['name']}. Reply with only the spoken line, at most 35 words, no quotes."
    )


def decide_prompt(agent, opp, r, total, pot, convo):
    return (
        f"{rules_for(agent['name'], opp['name'])}\n"
        f"This is round {r} of {total}. The pot is ${pot}. Standings: {standings()}.\n"
        f"{history_text()}\n"
        f"The negotiation just ended:\n{convo_text(convo)}\n"
        f"Weigh {opp['name']}'s promises against their track record and commit your secret choice.\n"
        "Reply in exactly this format:\nDECISION: SPLIT or STEAL\nREASON: one short sentence"
    )


def parse_decision(text):
    m = re.search(r"DECISION:\s*(SPLIT|STEAL)", text, re.I)
    d = m.group(1).upper() if m else ("STEAL" if re.search(r"\bSTEAL\b", text, re.I) else "SPLIT")
    m = re.search(r"REASON:\s*(.+)", text, re.I)
    reason = m.group(1).strip() if m else text.strip()[:120]
    return d, reason


def payout(pot, decisions):
    a, b = AGENTS[0]["name"], AGENTS[1]["name"]
    if decisions[a] == "SPLIT" and decisions[b] == "SPLIT":
        return {a: pot // 2, b: pot // 2}, "BOTH SPLIT — the pot is shared"
    if decisions[a] == "STEAL" and decisions[b] == "STEAL":
        return {a: 0, b: 0}, "BOTH STEAL — nobody gets a cent"
    thief = a if decisions[a] == "STEAL" else b
    mark = b if thief == a else a
    return {thief: pot, mark: 0}, f"{thief} STEALS the whole pot from {mark}"


def play(total_rounds):
    rounds_history.clear()
    with lock:
        state.update(status="running", round=0, total_rounds=total_rounds, pot=0,
                     totals={a["name"]: 0 for a in AGENTS}, log=[])
    for r in range(1, total_rounds + 1):
        pot = random.choice([80, 100, 120, 150, 200, 250])
        with lock:
            state["round"] = r
            state["pot"] = pot
        emit({"type": "round", "round": r, "pot": pot})
        convo = []
        order = [(AGENTS[0], AGENTS[1]), (AGENTS[1], AGENTS[0])]
        for _ in range(2):
            for agent, opp in order:
                emit({"type": "thinking", "agent": agent["name"]})
                line = run_agent(agent, talk_prompt(agent, opp, r, total_rounds, pot, convo))
                line = line.strip().strip('"')
                convo.append((agent["name"], line))
                emit({"type": "msg", "agent": agent["name"], "text": line})
        decisions, reasons = {}, {}
        for agent, opp in order:
            emit({"type": "thinking", "agent": agent["name"]})
            raw = run_agent(agent, decide_prompt(agent, opp, r, total_rounds, pot, convo))
            decisions[agent["name"]], reasons[agent["name"]] = parse_decision(raw)
        pay, outcome = payout(pot, decisions)
        with lock:
            for n, p in pay.items():
                state["totals"][n] += p
        rounds_history.append({"round": r, "pot": pot, "decisions": decisions, "payout": pay})
        emit({"type": "reveal", "round": r, "pot": pot, "decisions": decisions,
              "reasons": reasons, "payout": pay, "outcome": outcome})
    totals = state["totals"]
    best = max(totals.values())
    winners = [n for n, v in totals.items() if v == best]
    verdict = f"{winners[0]} wins with ${best} banked" if len(winners) == 1 else f"Dead tie at ${best}"
    emit({"type": "end", "verdict": verdict, "totals": dict(totals)})
    with lock:
        state["status"] = "done"


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def send(self, code, body, ctype):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/state":
            with lock:
                body = json.dumps(state).encode()
            self.send(200, body, "application/json")
            return
        path = "index.html"
        try:
            with open(os.path.join(os.path.dirname(__file__), path), "rb") as f:
                self.send(200, f.read(), "text/html")
        except OSError:
            self.send(404, b"not found", "text/plain")

    def do_POST(self):
        if self.path.startswith("/start"):
            with lock:
                busy = state["status"] == "running"
            if not busy:
                m = re.search(r"rounds=(\d+)", self.path)
                n = min(9, max(1, int(m.group(1)) if m else 5))
                threading.Thread(target=play, args=(n,), daemon=True).start()
            self.send(200, b'{"ok":true}', "application/json")
        else:
            self.send(404, b"not found", "text/plain")


if __name__ == "__main__":
    server = http.server.ThreadingHTTPServer(("127.0.0.1", PORT), Handler)
    print(f"Split or Steal on http://localhost:{PORT}")
    server.serve_forever()
