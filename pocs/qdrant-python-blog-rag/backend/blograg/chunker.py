def chunk_words(text: str, size: int = 220, overlap: int = 40) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = size - overlap
    last_start = max(len(words) - overlap, 1)
    return [" ".join(words[start : start + size]) for start in range(0, last_start, step)]
