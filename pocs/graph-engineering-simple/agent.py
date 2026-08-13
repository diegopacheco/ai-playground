import subprocess


def process_runner(command):
    result = subprocess.run(command, stdin=subprocess.DEVNULL, capture_output=True, text=True, check=False)
    if result.returncode != 0:
        output = f"{result.stdout}{result.stderr}".strip()
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {output}")
    return result.stdout.strip()


class Agent:
    def __init__(self, provider, runner=process_runner):
        self.provider = provider
        self.runner = runner

    def call(self, model, prompt, args=None):
        if not model.strip():
            raise ValueError("model is required")
        if not prompt.strip():
            raise ValueError("prompt is required")
        return self.runner(self.command(model, prompt, list(args or [])))

    def command(self, model, prompt, args):
        commands = {
            "claude": ["claude", "-p", "--model", model, *args, prompt],
            "codex": ["codex", "exec", "--model", model, *args, prompt],
            "agy": ["agy", "-p", "--model", model, *args, prompt],
            "ollama": ["ollama", "run", *args, model, prompt],
        }
        if self.provider not in commands:
            raise ValueError(f"Unknown provider: {self.provider}")
        return commands[self.provider]
