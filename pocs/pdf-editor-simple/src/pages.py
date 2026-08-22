def parse_pages(spec, total):
    selected = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            head, tail = part.split("-", 1)
            start, end = _number(head), _number(tail)
        else:
            start = end = _number(part)
        if start > end:
            raise ValueError(f"range '{part}' goes backwards")
        for page in range(start, end + 1):
            if page < 1 or page > total:
                raise ValueError(f"page {page} is outside 1-{total}")
            if page not in selected:
                selected.append(page)
    if not selected:
        raise ValueError("no pages selected")
    return selected


def invert_pages(selected, total):
    return [page for page in range(1, total + 1) if page not in selected]


def _number(text):
    text = text.strip()
    if not text.isdigit():
        raise ValueError(f"'{text}' is not a page number")
    return int(text)
