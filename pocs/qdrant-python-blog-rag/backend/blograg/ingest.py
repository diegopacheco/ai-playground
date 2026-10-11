from blograg import config
from blograg.blogger import Post, fetch_posts
from blograg.chunker import chunk_words
from blograg.ollama import DOCUMENT_PREFIX, Ollama
from blograg.store import Store

BATCH = 32


def embed_text(post: Post, chunk: str) -> str:
    return f"{post.title}\n{chunk}"


def flush(store: Store, llm: Ollama, rows: list[tuple[Post, int, str]]) -> None:
    if not rows:
        return
    vectors = llm.embed([embed_text(post, text) for post, _, text in rows], DOCUMENT_PREFIX)
    store.upsert_chunks(rows, vectors)
    rows.clear()


def main() -> None:
    settings = config.load()
    store = Store(settings.qdrant_url, settings.collection)
    store.ensure_collection()
    llm = Ollama(settings.ollama_url, settings.embed_model, settings.chat_model)
    rows: list[tuple[Post, int, str]] = []
    posts = 0
    for post in fetch_posts(settings.blog_url):
        posts += 1
        for number, chunk in enumerate(chunk_words(post.text)):
            rows.append((post, number, chunk))
            if len(rows) >= BATCH:
                flush(store, llm, rows)
        if posts % 50 == 0:
            print(f"indexed {posts} posts", flush=True)
    flush(store, llm, rows)
    print(f"done: {store.count_posts()} posts, {store.count_chunks()} chunks in {settings.collection}", flush=True)


if __name__ == "__main__":
    main()
