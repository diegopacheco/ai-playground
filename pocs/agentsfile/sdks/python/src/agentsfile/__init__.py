from pathlib import Path

from agentsfile.errors import AgentsfileError
from agentsfile.lexer import tokenize
from agentsfile.model import Agent, Agentsfile, Command, Mcp, Plugin, Profile, Skill, SkillFile, Transport
from agentsfile.parser import parse_statements
from agentsfile.schema import build

__all__ = [
    "Agent",
    "Agentsfile",
    "AgentsfileError",
    "Command",
    "Mcp",
    "Plugin",
    "Profile",
    "Skill",
    "SkillFile",
    "Transport",
    "load",
    "parse",
]


def parse(text: str) -> Agentsfile:
    return build(parse_statements(tokenize(text)))


def load(path: str | Path) -> Agentsfile:
    return parse(Path(path).read_text(encoding="utf-8"))
