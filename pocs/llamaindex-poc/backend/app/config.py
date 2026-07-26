import os
from dataclasses import dataclass
from pathlib import Path

ROOT: Path = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Config:
    data_dir: Path
    uploads_dir: Path
    index_dir: Path
    annotated_dir: Path
    agents_file: Path
    rust_binary: Path
    gguf_model: Path
    ollama_host: str
    embed_model: str
    llm_model: str
    chunk_size: int
    chunk_overlap: int
    top_k: int
    request_timeout: float

    def ensure_dirs(self) -> None:
        for path in (self.data_dir, self.uploads_dir, self.index_dir, self.annotated_dir):
            path.mkdir(parents=True, exist_ok=True)


def load_config() -> Config:
    data_dir = Path(os.getenv("DATA_DIR", ROOT / "data")).resolve()
    default_gguf = ROOT.parent / "rust-llama-local" / "models" / "llama-3.gguf"
    config = Config(
        data_dir=data_dir,
        uploads_dir=data_dir / "uploads",
        index_dir=data_dir / "index",
        annotated_dir=data_dir / "annotated",
        agents_file=data_dir / "agents.json",
        rust_binary=Path(
            os.getenv("PDFLLAMA_BIN", ROOT / "rust" / "target" / "release" / "pdfllama")
        ),
        gguf_model=Path(os.getenv("PDFLLAMA_MODEL", default_gguf)),
        ollama_host=os.getenv("OLLAMA_HOST", "http://localhost:11434"),
        embed_model=os.getenv("EMBED_MODEL", "nomic-embed-text"),
        llm_model=os.getenv("LLM_MODEL", "llama3.2"),
        chunk_size=int(os.getenv("CHUNK_SIZE", "512")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "64")),
        top_k=int(os.getenv("TOP_K", "5")),
        request_timeout=float(os.getenv("OLLAMA_TIMEOUT", "300")),
    )
    config.ensure_dirs()
    return config


config: Config = load_config()
