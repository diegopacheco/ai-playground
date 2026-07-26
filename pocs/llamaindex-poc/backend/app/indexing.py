from __future__ import annotations

import threading
from pathlib import Path
from typing import Any

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
    load_index_from_storage,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import NodeWithScore
from llama_index.core.vector_stores import FilterOperator, MetadataFilter, MetadataFilters
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.readers.file import PDFReader

from .config import config
from .storage import DocumentRecord, store, utc_now


def configure_settings() -> None:
    Settings.embed_model = OllamaEmbedding(
        model_name=config.embed_model,
        base_url=config.ollama_host,
    )
    Settings.llm = Ollama(
        model=config.llm_model,
        base_url=config.ollama_host,
        request_timeout=config.request_timeout,
    )
    Settings.node_parser = SentenceSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
    )


def page_doc_id(doc_id: str, page: int) -> str:
    return f"{doc_id}::p{page}"


class IndexService:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._index: VectorStoreIndex | None = None

    def _persisted(self) -> bool:
        return (config.index_dir / "docstore.json").is_file()

    def index(self) -> VectorStoreIndex:
        with self._lock:
            if self._index is None:
                if self._persisted():
                    storage_context = StorageContext.from_defaults(
                        persist_dir=str(config.index_dir)
                    )
                    loaded = load_index_from_storage(storage_context)
                    if not isinstance(loaded, VectorStoreIndex):
                        raise RuntimeError("persisted index is not a vector index")
                    self._index = loaded
                else:
                    self._index = VectorStoreIndex([])
            return self._index

    def _persist(self) -> None:
        self.index().storage_context.persist(persist_dir=str(config.index_dir))

    def is_empty(self) -> bool:
        return len(store.list()) == 0

    def ingest(self, doc_id: str, file_name: str, path: Path) -> DocumentRecord:
        pages: list[Document] = PDFReader(return_full_document=False).load_data(file=path)
        if not pages:
            raise ValueError("no pages found in this pdf")

        chars = 0
        index = self.index()
        for number, page in enumerate(pages, start=1):
            page.id_ = page_doc_id(doc_id, number)
            page.metadata = {
                "doc_id": doc_id,
                "file_name": file_name,
                "page": number,
            }
            page.excluded_embed_metadata_keys = ["doc_id"]
            page.excluded_llm_metadata_keys = ["doc_id"]
            chars += len(page.text)
            index.insert(page)

        self._persist()
        chunks = sum(
            1
            for node in index.docstore.docs.values()
            if node.metadata.get("doc_id") == doc_id
        )
        record = DocumentRecord(
            doc_id=doc_id,
            file_name=file_name,
            size_bytes=path.stat().st_size,
            pages=len(pages),
            chunks=chunks,
            chars=chars,
            ingested_at=utc_now(),
        )
        store.register(record)
        return record

    def delete(self, record: DocumentRecord) -> None:
        index = self.index()
        for number in range(1, record.pages + 1):
            index.delete_ref_doc(page_doc_id(record.doc_id, number), delete_from_docstore=True)
        self._persist()

    def _filters(self, doc_ids: list[str] | None) -> MetadataFilters | None:
        if not doc_ids:
            return None
        return MetadataFilters(
            filters=[
                MetadataFilter(key="doc_id", value=doc_id, operator=FilterOperator.EQ)
                for doc_id in doc_ids
            ],
            condition="or",
        )

    def retrieve(
        self, question: str, top_k: int, doc_ids: list[str] | None = None
    ) -> list[NodeWithScore]:
        if self.is_empty():
            return []
        retriever = self.index().as_retriever(
            similarity_top_k=top_k,
            filters=self._filters(doc_ids),
        )
        return retriever.retrieve(question)

    def scan(self, term: str, doc_ids: list[str] | None = None) -> list[NodeWithScore]:
        needle = term.lower()
        allowed = set(doc_ids or [])
        hits: list[NodeWithScore] = []
        for node in self.index().docstore.docs.values():
            if allowed and node.metadata.get("doc_id") not in allowed:
                continue
            text = node.get_content()
            count = text.lower().count(needle)
            if count:
                hits.append(NodeWithScore(node=node, score=float(count)))
        return sorted(hits, key=lambda hit: hit.score or 0.0, reverse=True)

    def stats(self) -> dict[str, Any]:
        records = store.list()
        return {
            "documents": len(records),
            "pages": sum(record.pages for record in records),
            "chunks": sum(record.chunks for record in records),
            "chars": sum(record.chars for record in records),
            "embed_model": config.embed_model,
            "llm_model": config.llm_model,
            "chunk_size": config.chunk_size,
            "chunk_overlap": config.chunk_overlap,
        }


service: IndexService = IndexService()
