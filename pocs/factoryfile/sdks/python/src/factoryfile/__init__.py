from pathlib import Path

from factoryfile.errors import FactoryfileError
from factoryfile.lexer import tokenize
from factoryfile.model import Agent, Factory, Factoryfile, Loop, Stage
from factoryfile.parser import parse_statements
from factoryfile.schema import build

__all__ = [
    "Agent",
    "Factory",
    "Factoryfile",
    "FactoryfileError",
    "Loop",
    "Stage",
    "load",
    "parse",
]


def parse(text: str) -> Factoryfile:
    return build(parse_statements(tokenize(text)))


def load(path: str | Path) -> Factoryfile:
    return parse(Path(path).read_text(encoding="utf-8"))
