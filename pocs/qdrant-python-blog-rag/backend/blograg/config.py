import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Config:
    api_port: int
    qdrant_url: str
    ollama_url: str
    embed_model: str
    chat_model: str
    blog_url: str
    collection: str


def load() -> Config:
    return Config(
        api_port=int(os.environ.get("API_PORT", "8101")),
        qdrant_url=os.environ.get("QDRANT_URL", "http://localhost:6333"),
        ollama_url=os.environ.get("OLLAMA_URL", "http://localhost:11434"),
        embed_model=os.environ.get("EMBED_MODEL", "nomic-embed-text"),
        chat_model=os.environ.get("CHAT_MODEL", "llama3.2"),
        blog_url=os.environ.get("BLOG_URL", "https://diego-pacheco.blogspot.com"),
        collection=os.environ.get("COLLECTION", "blog_posts"),
    )
