import datetime
import json
import os
import re
import subprocess
import threading

from flask import Flask, jsonify, render_template, request

import catalog

BASE = os.path.dirname(os.path.abspath(__file__))
GAMES_FILE = os.path.join(BASE, "games.json")
WIP_FILE = os.path.join(BASE, "wip.json")
CONFIG_FILE = os.path.join(BASE, "config.json")
LOCK = threading.Lock()

AGENTS = {
    "claude": ["claude", "-p"],
    "codex": ["codex", "exec"],
    "agy": ["agy", "-p"],
}

app = Flask(__name__)


def load_games():
    with open(GAMES_FILE) as f:
        return json.load(f)


def save_games(games):
    with open(GAMES_FILE, "w") as f:
        json.dump(games, f, indent=2)


def load_wip():
    if os.path.exists(WIP_FILE):
        with open(WIP_FILE) as f:
            return json.load(f)
    return []


def save_wip(wip):
    with open(WIP_FILE, "w") as f:
        json.dump(wip, f, indent=2)


def load_config():
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE) as f:
            return json.load(f)
    return {"agent": "claude"}


def save_config(config):
    with open(CONFIG_FILE, "w") as f:
        json.dump(config, f, indent=2)


def agent_cover_url(agent, name, console):
    prompt = (
        'Search the web for the official box art or cover image of the video game "'
        + name + '" on ' + console +
        ". Answer with only one direct image URL ending in .jpg, .jpeg, .png or .webp and nothing else."
    )
    try:
        result = subprocess.run(
            AGENTS[agent] + [prompt], capture_output=True, text=True, timeout=300
        )
        match = re.search(
            r"https?://\S+?\.(?:jpg|jpeg|png|webp)\b", result.stdout, re.IGNORECASE
        )
        if match:
            return match.group(0)
    except Exception:
        pass
    return None


def fetch_cover(agent, name, console, slug, idx):
    url = agent_cover_url(agent, name, console)
    source = "agent"
    if not url:
        try:
            url = catalog.wiki_thumb(name + " video game")
            source = "wikipedia"
        except Exception:
            url = None
    if url:
        try:
            fname = slug + catalog.ext_for(url)
            catalog.download(url, os.path.join(catalog.COVERS, fname))
            return "covers/" + fname, source
        except Exception:
            pass
    fname = slug + ".svg"
    catalog.placeholder(name, os.path.join(catalog.COVERS, fname), idx)
    return "covers/" + fname, "placeholder"


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/games")
def api_games():
    return jsonify(load_games())


@app.route("/api/games", methods=["POST"])
def api_add_game():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    console = data.get("console")
    if not name or console not in ("PS5", "Switch", "Steam"):
        return jsonify({"error": "name and console (PS5, Switch or Steam) are required"}), 400
    with LOCK:
        games = load_games()
        slug = catalog.slugify(name)
        if any(g["id"] == slug for g in games):
            return jsonify({"error": name + " is already in the catalog"}), 409
        agent = load_config().get("agent", "claude")
        cover, source = fetch_cover(agent, name, console, slug, len(games))
        year = data.get("year") or datetime.date.today().year
        order = max((g["order"] for g in games), default=0) + 1
        game = {"id": slug, "name": name, "console": console,
                "year": int(year), "order": order, "cover": cover}
        games.append(game)
        save_games(games)
    return jsonify({"game": game, "source": source, "agent": agent}), 201


@app.route("/api/games/<game_id>", methods=["DELETE"])
def api_delete_game(game_id):
    with LOCK:
        games = load_games()
        game = next((g for g in games if g["id"] == game_id), None)
        if not game:
            return jsonify({"error": "game not found"}), 404
        games = [g for g in games if g["id"] != game_id]
        save_games(games)
        cover_path = os.path.join(BASE, "static", game["cover"])
        if os.path.exists(cover_path):
            os.remove(cover_path)
    return jsonify({"deleted": game_id})


@app.route("/api/wip")
def api_wip():
    return jsonify(load_wip())


@app.route("/api/wip", methods=["POST"])
def api_add_wip():
    data = request.get_json(force=True)
    name = (data.get("name") or "").strip()
    console = data.get("console")
    if not name or console not in ("PS5", "Switch", "Steam"):
        return jsonify({"error": "name and console (PS5, Switch or Steam) are required"}), 400
    with LOCK:
        wip = load_wip()
        slug = catalog.slugify(name)
        if any(w["id"] == slug for w in wip):
            return jsonify({"error": name + " is already in progress"}), 409
        if any(g["id"] == slug for g in load_games()):
            return jsonify({"error": name + " is already finished in the library"}), 409
        agent = load_config().get("agent", "claude")
        cover, source = fetch_cover(agent, name, console, slug, len(wip))
        game = {"id": slug, "name": name, "console": console, "cover": cover}
        wip.append(game)
        save_wip(wip)
    return jsonify({"game": game, "source": source, "agent": agent}), 201


@app.route("/api/wip/<game_id>", methods=["DELETE"])
def api_delete_wip(game_id):
    with LOCK:
        wip = load_wip()
        game = next((w for w in wip if w["id"] == game_id), None)
        if not game:
            return jsonify({"error": "game not found"}), 404
        save_wip([w for w in wip if w["id"] != game_id])
        cover_path = os.path.join(BASE, "static", game["cover"])
        if os.path.exists(cover_path):
            os.remove(cover_path)
    return jsonify({"deleted": game_id})


@app.route("/api/wip/<game_id>/finish", methods=["POST"])
def api_finish_wip(game_id):
    with LOCK:
        wip = load_wip()
        game = next((w for w in wip if w["id"] == game_id), None)
        if not game:
            return jsonify({"error": "game not found"}), 404
        games = load_games()
        if any(g["id"] == game_id for g in games):
            return jsonify({"error": game["name"] + " is already in the library"}), 409
        finished = {"id": game["id"], "name": game["name"], "console": game["console"],
                    "year": datetime.date.today().year,
                    "order": max((g["order"] for g in games), default=0) + 1,
                    "cover": game["cover"]}
        games.append(finished)
        save_games(games)
        save_wip([w for w in wip if w["id"] != game_id])
    return jsonify({"game": finished})


@app.route("/api/config")
def api_config():
    return jsonify(load_config())


@app.route("/api/config", methods=["POST"])
def api_set_config():
    data = request.get_json(force=True)
    agent = data.get("agent")
    if agent not in AGENTS:
        return jsonify({"error": "agent must be claude, codex or agy"}), 400
    config = {"agent": agent}
    save_config(config)
    return jsonify(config)


if __name__ == "__main__":
    port = int(os.environ.get("GAME_STAND_PORT", "5057"))
    app.run(host="127.0.0.1", port=port)
