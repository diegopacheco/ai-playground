from io import BytesIO

import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from pypdf.generic import DecodedStreamObject, DictionaryObject, FloatObject, NameObject

from textmap import page_runs

FONT_KEY = "/PDFEDITHELV"
ALPHA_KEY = "/PDFEDITALPHA"
PADDING = 1.0
COVER_MARKER = b"%PDFEDIT-COVER"


def read_runs(pdf_bytes, page_index):
    source = PdfReader(BytesIO(pdf_bytes)).pages[page_index]
    contents = _contents(source)
    fonts = source.get("/Resources", {}).get("/Font", {})
    document = pdfium.PdfDocument(pdf_bytes)
    page = document[page_index]
    width, height = page.get_width(), page.get_height()
    runs = page_runs(page, contents, dict(fonts), covered_boxes(contents))
    rotation = page.get_rotation()
    for run in runs:
        run["display"] = _displayed(run["box"], rotation, page)
    return width, height, runs


def _displayed(box, rotation, page):
    if not rotation:
        return list(box)
    unrotated_width = page.get_height() if rotation in (90, 270) else page.get_width()
    unrotated_height = page.get_width() if rotation in (90, 270) else page.get_height()
    corners = [(box[0], box[1]), (box[2], box[3])]
    moved = [_turn(x, y, rotation, unrotated_width, unrotated_height) for x, y in corners]
    xs = [point[0] for point in moved]
    ys = [point[1] for point in moved]
    return [min(xs), min(ys), max(xs), max(ys)]


def to_page(point, rotation, width, height):
    x, y = point
    if not rotation:
        return x, y
    if rotation == 90:
        return width - y, x
    if rotation == 180:
        return width - x, height - y
    return y, height - x


def page_shape(pdf_bytes, page_index):
    page = pdfium.PdfDocument(pdf_bytes)[page_index]
    rotation = page.get_rotation()
    if rotation in (90, 270):
        return rotation, page.get_height(), page.get_width()
    return rotation, page.get_width(), page.get_height()


def _turn(x, y, rotation, width, height):
    if rotation == 90:
        return y, width - x
    if rotation == 180:
        return width - x, height - y
    return height - y, x


def covered_boxes(contents):
    covers = []
    for line in contents.split(b"\n"):
        if line.startswith(COVER_MARKER):
            fields = line.split()
            text = bytes.fromhex(fields[5].decode("latin-1")).decode("utf-8") if len(fields) > 5 else ""
            covers.append(([float(value) for value in fields[1:5]], text))
    return covers


def apply_edits(pdf_bytes, page_index, edits):
    return apply_changes(pdf_bytes, page_index, {
        run_id: {"text": text, "dx": 0.0, "dy": 0.0} for run_id, text in edits.items()
    })


def apply_changes(pdf_bytes, page_index, changes):
    width, height, runs = read_runs(pdf_bytes, page_index)
    by_id = {run["id"]: run for run in runs}
    writer = PdfWriter(clone_from=BytesIO(pdf_bytes))
    page = writer.pages[page_index]
    contents = _contents(page)

    replacements = []
    removals = []
    overlay = []
    report = []
    background = None

    for run_id, change in changes.items():
        run = by_id.get(int(run_id))
        if run is None:
            raise ValueError(f"page {page_index + 1} has no text run {run_id}")
        new_text = change["text"] if change.get("text") is not None else run["text"]
        moved = bool(change.get("dx") or change.get("dy"))
        if moved:
            run = dict(run, x=run["x"] + change["dx"], y=run["y"] + change["dy"],
                       box=[run["box"][0] + change["dx"], run["box"][1] + change["dy"],
                            run["box"][2] + change["dx"], run["box"][3] + change["dy"]])
        if not moved and run["mode"] == "inplace" and _encodable(run, new_text):
            replacements.append((run, new_text))
            report.append({"run": run["id"], "mode": "inplace"})
        else:
            missing = "" if _keeps_typeface(run, new_text) else _undrawable(new_text)
            if missing:
                raise ValueError(
                    f"the fallback font cannot draw {missing}, so this line cannot be redrawn"
                )
            if run["operations"]:
                removals.extend(run["operations"])
                overlay.append((run, new_text, False))
                report.append({
                    "run": run["id"],
                    "mode": "replaced" if _keeps_typeface(run, new_text) else "redrawn",
                })
            else:
                if background is None:
                    background = _background(pdf_bytes, page_index)
                overlay.append((run, new_text, True))
                report.append({"run": run["id"], "mode": "redraw"})

    for run, new_text in sorted(replacements, key=lambda pair: pair[0]["operation"]["start"], reverse=True):  # noqa: E501
        operation = run["operation"]
        codes = _encode(run, new_text)
        contents = contents[:operation["start"]] + _literal(codes) + b" Tj" + contents[operation["end"]:]

    for operation in sorted(removals, key=lambda item: item["start"], reverse=True):
        contents = contents[:operation["start"]] + contents[operation["end"]:]

    if overlay:
        contents = b"q\n" + contents + b"\nQ\n" + _overlay(overlay, background)
        if any(not _keeps_typeface(run, text) for run, text, _ in overlay):
            _ensure_font(writer, page)

    page.replace_contents(DecodedStreamObject.initialize_from_dictionary({"__streamdata__": contents, "/Length": len(contents)}))
    output = BytesIO()
    writer.write(output)
    return output.getvalue(), report


def bake(pdf_bytes, page_index, notes):
    if not notes:
        return pdf_bytes
    writer = PdfWriter(clone_from=BytesIO(pdf_bytes))
    page = writer.pages[page_index]
    contents = _contents(page)
    drawing = [b"q"]
    wants_alpha = False
    for note in notes:
        if note["kind"] == "highlight":
            wants_alpha = True
            red, green, blue = note["color"]
            drawing.append(
                f"q {ALPHA_KEY} gs {red:.4f} {green:.4f} {blue:.4f} rg "
                f"{note['x']:.2f} {note['y']:.2f} {note['width']:.2f} {note['height']:.2f} re f Q".encode("latin-1")
            )
        elif note["text"]:
            red, green, blue = note["color"]
            drawing.append(
                b"BT " + f"{FONT_KEY} {note['size']:.2f} Tf {red:.4f} {green:.4f} {blue:.4f} rg "
                f"{note['x']:.2f} {note['y']:.2f} Td ".encode("latin-1")
                + _literal(note["text"].encode("cp1252", "replace")) + b" Tj ET"
            )
    drawing.append(b"Q")
    contents = b"q\n" + contents + b"\nQ\n" + b"\n".join(drawing)
    _ensure_font(writer, page)
    if wants_alpha:
        _ensure_alpha(writer, page)
    page.replace_contents(DecodedStreamObject.initialize_from_dictionary(
        {"__streamdata__": contents, "/Length": len(contents)}))
    output = BytesIO()
    writer.write(output)
    return output.getvalue()


def _ensure_alpha(writer, page):
    resources = page.get("/Resources")
    states = resources.get("/ExtGState")
    if states is None:
        states = DictionaryObject()
        resources[NameObject("/ExtGState")] = states
    if ALPHA_KEY not in states:
        state = DictionaryObject()
        state.update({
            NameObject("/Type"): NameObject("/ExtGState"),
            NameObject("/ca"): FloatObject(0.38),
            NameObject("/BM"): NameObject("/Multiply"),
        })
        states[NameObject(ALPHA_KEY)] = writer._add_object(state)


def _overlay(overlay, background):
    red, green, blue = background or (1.0, 1.0, 1.0)
    parts = [b"q"]
    for run, new_text, cover in overlay:
        left, bottom, right, top = run["box"]
        if not cover:
            parts.append(_draw(run, new_text))
            continue
        parts.append(
            f"{COVER_MARKER.decode()} {left - PADDING:.2f} {bottom - PADDING:.2f} "
            f"{right + PADDING:.2f} {top + PADDING:.2f} "
            f"{new_text.encode('utf-8').hex()}".encode("latin-1")
        )
        parts.append(
            f"{red:.4f} {green:.4f} {blue:.4f} rg "
            f"{left - PADDING:.2f} {bottom - PADDING:.2f} "
            f"{right - left + 2 * PADDING:.2f} {top - bottom + 2 * PADDING:.2f} re f".encode("latin-1")
        )
        if new_text:
            parts.append(_draw(run, new_text))
    parts.append(b"Q")
    return b"\n".join(parts)


def _draw(run, new_text):
    colour = " ".join(f"{channel:.4f}" for channel in run["color"])
    resource, codes = _typeface(run, new_text)
    return (
        b"BT " + f"{resource} {run['draw_size']:.2f} Tf {colour} rg "
        f"{run['x']:.2f} {run['y']:.2f} Td ".encode("latin-1")
        + _literal(codes) + b" Tj ET"
    )


def _typeface(run, new_text):
    resource = run.get("font_resource")
    if resource and run.get("standard_font"):
        codes = _latin(new_text)
        if codes is not None:
            return resource, codes
    alphabet = run.get("alphabet") or {}
    if resource and new_text and all(character in alphabet for character in new_text):
        return resource, b"".join(alphabet[character] for character in new_text)
    return FONT_KEY, _latin(new_text) or b""


def _latin(text):
    try:
        return text.encode("cp1252")
    except UnicodeEncodeError:
        return None


def _keeps_typeface(run, new_text):
    return _typeface(run, new_text)[0] != FONT_KEY


def _ensure_font(writer, page):
    resources = page.get("/Resources")
    fonts = resources.get("/Font")
    if fonts is None:
        fonts = DictionaryObject()
        resources[NameObject("/Font")] = fonts
    if FONT_KEY not in fonts:
        font = DictionaryObject()
        font.update({
            NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"),
            NameObject("/BaseFont"): NameObject("/Helvetica"),
            NameObject("/Encoding"): NameObject("/WinAnsiEncoding"),
        })
        fonts[NameObject(FONT_KEY)] = writer._add_object(font)


def _background(pdf_bytes, page_index):
    page = pdfium.PdfDocument(pdf_bytes)[page_index]
    bitmap = page.render(scale=0.4, rev_byteorder=True)
    pixels = bytes(bitmap.buffer)
    counts = {}
    for y in range(0, bitmap.height, 3):
        row = y * bitmap.stride
        for x in range(0, bitmap.width, 3):
            pixel = pixels[row + x * 3:row + x * 3 + 3]
            counts[pixel] = counts.get(pixel, 0) + 1
    common = max(counts, key=counts.get)
    return tuple(round(channel / 255, 4) for channel in common)


def _undrawable(text):
    missing = []
    for character in text:
        try:
            character.encode("cp1252")
        except UnicodeEncodeError:
            if character not in missing:
                missing.append(character)
    return " ".join(repr(character) for character in missing)


def _encodable(run, text):
    if run.get("full_charset"):
        try:
            text.encode("cp1252")
            return True
        except UnicodeEncodeError:
            return False
    return all(character in run.get("encoding", {}) for character in text)


def _encode(run, text):
    if run.get("full_charset"):
        return text.encode("cp1252")
    return b"".join(run["encoding"][character] for character in text)


def _literal(data):
    escaped = data.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
    return b"(" + escaped + b")"


def _contents(page):
    contents = page.get_contents()
    return contents.get_data() if contents is not None else b""
