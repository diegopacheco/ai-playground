import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient, models

from blograg.blogger import Post

VECTOR_SIZE = 768


@dataclass(frozen=True)
class Hit:
    url: str
    title: str
    published: str
    labels: list[str]
    score: float
    text: str


@dataclass(frozen=True)
class Facet:
    name: str
    count: int


def point_id(url: str, chunk: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, f"{url}#{chunk}"))


def build_filter(label: str | None, year: int | None) -> models.Filter | None:
    must: list[models.Condition] = []
    if label:
        must.append(models.FieldCondition(key="labels", match=models.MatchValue(value=label)))
    if year:
        must.append(models.FieldCondition(key="year", match=models.MatchValue(value=year)))
    return models.Filter(must=must) if must else None


def to_hit(payload: dict[str, Any] | None, score: float) -> Hit:
    data = payload or {}
    return Hit(
        url=data["url"],
        title=data["title"],
        published=data["published"],
        labels=data["labels"],
        score=round(score, 4),
        text=data["text"],
    )


class Store:
    def __init__(self, url: str, collection: str) -> None:
        self.client = QdrantClient(url=url)
        self.collection = collection

    def ensure_collection(self) -> None:
        if self.client.collection_exists(self.collection):
            return
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(size=VECTOR_SIZE, distance=models.Distance.COSINE),
        )
        self.client.create_payload_index(self.collection, "labels", models.PayloadSchemaType.KEYWORD)
        self.client.create_payload_index(self.collection, "year", models.PayloadSchemaType.INTEGER)
        self.client.create_payload_index(self.collection, "chunk", models.PayloadSchemaType.INTEGER)

    def upsert_chunks(self, rows: Sequence[tuple[Post, int, str]], vectors: Sequence[list[float]]) -> None:
        points = [
            models.PointStruct(
                id=point_id(post.url, chunk),
                vector=vector,
                payload={
                    "url": post.url,
                    "title": post.title,
                    "published": post.published,
                    "year": post.year,
                    "labels": list(post.labels),
                    "chunk": chunk,
                    "text": text,
                },
            )
            for (post, chunk, text), vector in zip(rows, vectors, strict=True)
        ]
        self.client.upsert(self.collection, points=points, wait=True)

    def search_posts(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]:
        result = self.client.query_points_groups(
            self.collection,
            query=vector,
            group_by="url",
            limit=limit,
            group_size=1,
            query_filter=build_filter(label, year),
            with_payload=True,
        )
        return [to_hit(group.hits[0].payload, group.hits[0].score) for group in result.groups]

    def search_chunks(self, vector: list[float], limit: int, label: str | None, year: int | None) -> list[Hit]:
        result = self.client.query_points(
            self.collection,
            query=vector,
            limit=limit,
            query_filter=build_filter(label, year),
            with_payload=True,
        )
        return [to_hit(point.payload, point.score) for point in result.points]

    def count_chunks(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def count_posts(self) -> int:
        first_chunks = models.Filter(must=[models.FieldCondition(key="chunk", match=models.MatchValue(value=0))])
        return self.client.count(self.collection, count_filter=first_chunks, exact=True).count

    def facet(self, key: str, limit: int) -> list[Facet]:
        result = self.client.facet(self.collection, key=key, limit=limit)
        return [Facet(name=str(hit.value), count=hit.count) for hit in result.hits]
