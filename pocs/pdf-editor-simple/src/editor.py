from pypdf import PdfReader, PdfWriter

from pages import invert_pages, parse_pages


def info(path):
    reader = PdfReader(path)
    meta = reader.metadata or {}
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        box = page.mediabox
        pages.append((index, round(float(box.width)), round(float(box.height)), page.rotation))
    return {
        "title": meta.get("/Title", ""),
        "producer": meta.get("/Producer", ""),
        "encrypted": reader.is_encrypted,
        "pages": pages,
    }


def merge(paths, out_path):
    writer = PdfWriter()
    for path in paths:
        for page in PdfReader(path).pages:
            writer.add_page(page)
    return _write(writer, out_path)


def split(path, out_dir):
    reader = PdfReader(path)
    written = []
    for index, page in enumerate(reader.pages, start=1):
        writer = PdfWriter()
        writer.add_page(page)
        written.append(_write(writer, f"{out_dir}/page-{index:03d}.pdf"))
    return written


def extract(path, spec, out_path):
    reader = PdfReader(path)
    selected = parse_pages(spec, len(reader.pages))
    return _write(_pick(reader, selected), out_path)


def delete(path, spec, out_path):
    reader = PdfReader(path)
    total = len(reader.pages)
    kept = invert_pages(parse_pages(spec, total), total)
    if not kept:
        raise ValueError("deleting every page would leave an empty document")
    return _write(_pick(reader, kept), out_path)


def rotate(path, spec, angle, out_path):
    if angle % 90:
        raise ValueError("angle must be a multiple of 90")
    reader = PdfReader(path)
    selected = parse_pages(spec, len(reader.pages))
    writer = PdfWriter()
    for index, page in enumerate(reader.pages, start=1):
        if index in selected:
            page.rotate(angle)
        writer.add_page(page)
    return _write(writer, out_path)


def text(path, spec):
    reader = PdfReader(path)
    total = len(reader.pages)
    selected = parse_pages(spec, total) if spec else list(range(1, total + 1))
    return [(page, reader.pages[page - 1].extract_text().strip()) for page in selected]


def _pick(reader, selected):
    writer = PdfWriter()
    for page in selected:
        writer.add_page(reader.pages[page - 1])
    return writer


def _write(writer, out_path):
    with open(out_path, "wb") as handle:
        writer.write(handle)
    return out_path
