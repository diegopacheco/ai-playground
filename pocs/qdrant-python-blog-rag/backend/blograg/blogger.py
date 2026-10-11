import json
from collections.abc import Iterator
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.request import urlopen

PAGE_SIZE = 150


@dataclass(frozen=True)
class Post:
    url: str
    title: str
    published: str
    labels: tuple[str, ...]
    text: str

    @property
    def year(self) -> int:
        return int(self.published[:4])


class _TextExtractor(HTMLParser):
    HIDDEN = frozenset({"script", "style"})

    def __init__(self) -> None:
        super().__init__()
        self.parts: list[str] = []
        self.hidden_depth = 0

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in self.HIDDEN:
            self.hidden_depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self.HIDDEN and self.hidden_depth:
            self.hidden_depth -= 1

    def handle_data(self, data: str) -> None:
        if not self.hidden_depth:
            self.parts.append(data)


def html_to_text(html: str) -> str:
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    return " ".join(" ".join(parser.parts).split())


def parse_entry(entry: dict[str, Any]) -> Post | None:
    links = [link["href"] for link in entry.get("link", []) if link.get("rel") == "alternate"]
    body = entry.get("content") or entry.get("summary") or {}
    text = html_to_text(body.get("$t", ""))
    if not links or not text:
        return None
    return Post(
        url=links[0],
        title=entry["title"]["$t"].strip(),
        published=entry["published"]["$t"],
        labels=tuple(category["term"] for category in entry.get("category", [])),
        text=text,
    )


def fetch_page(blog_url: str, start: int) -> list[dict[str, Any]]:
    url = f"{blog_url}/feeds/posts/default?alt=json&max-results={PAGE_SIZE}&start-index={start}"
    with urlopen(url, timeout=60) as response:
        feed: dict[str, Any] = json.load(response)["feed"]
    entries: list[dict[str, Any]] = feed.get("entry", [])
    return entries


def fetch_posts(blog_url: str) -> Iterator[Post]:
    start = 1
    while entries := fetch_page(blog_url, start):
        for entry in entries:
            post = parse_entry(entry)
            if post:
                yield post
        start += len(entries)
