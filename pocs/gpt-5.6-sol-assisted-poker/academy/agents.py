import shutil
import subprocess


PROVIDERS = {
    "claude": {"label": "Claude", "command": ["claude", "-p"]},
    "codex": {"label": "Codex", "command": ["codex", "exec", "--ephemeral", "--sandbox", "read-only"]},
    "agy": {"label": "Agy", "command": ["agy", "-p"]},
}


def fallback_action(chance, legal):
    if chance >= 68 and "raise" in legal:
        return "raise"
    if chance >= 42:
        return "call" if "call" in legal else "check"
    return "fold" if "fold" in legal else "check"


def agent_action(provider, context, chance, legal):
    config = PROVIDERS.get(provider, PROVIDERS["claude"])
    executable = shutil.which(config["command"][0])
    if not executable:
        return fallback_action(chance, legal), f"{config['label']} is unavailable, so the local poker policy acted."
    prompt = (
        "You are playing heads-up no-limit Texas Hold'em. "
        f"Situation: {context}. Your estimated equity is {chance}%. "
        f"Legal actions: {', '.join(legal)}. "
        "Reply with exactly one legal action and nothing else."
    )
    command = [executable, *config["command"][1:], prompt]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=45, check=False)
        output = result.stdout.strip().lower()
        for action in legal:
            if action in output:
                return action, f"{config['label']} chose {action}."
    except (OSError, subprocess.TimeoutExpired):
        pass
    action = fallback_action(chance, legal)
    return action, f"{config['label']} did not return a valid move, so the local poker policy chose {action}."

