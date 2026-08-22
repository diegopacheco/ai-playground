import ctypes

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from content import show_operations

LINE_TOLERANCE = 0.5
GAP_TOLERANCE = 0.9


def page_runs(page, contents, fonts=None, covers=()):
    textpage = page.get_textpage()
    characters = _characters(textpage)
    visible = _uncovered(characters, covers)
    runs = _group(characters, visible)
    _mark_editable(runs, characters, contents, fonts or {})
    return runs


def _uncovered(characters, covers):
    drawn_last = sum(len(text) for _, text in covers)
    original = len(characters) - drawn_last
    visible = []
    for index, character in enumerate(characters):
        left, bottom, right, top = character["box"]
        covered = index < original and any(
            box[0] - 0.5 <= left and box[1] - 0.5 <= bottom
            and box[2] + 0.5 >= right and box[3] + 0.5 >= top
            for box, _ in covers
        )
        if not covered:
            visible.append(index)
    return visible


def is_standard(font):
    if font is None:
        return False
    if font.get("/Subtype") not in ("/Type1", "/TrueType"):
        return False
    descriptor = font.get("/FontDescriptor")
    descriptor = descriptor.get_object() if descriptor is not None else None
    if descriptor is not None and any(key in descriptor for key in ("/FontFile", "/FontFile2", "/FontFile3")):
        return False
    encoding = font.get("/Encoding")
    encoding = encoding.get_object() if hasattr(encoding, "get_object") else encoding
    if isinstance(encoding, dict) and "/Differences" in encoding:
        return False
    return encoding in (None, "/WinAnsiEncoding", "/StandardEncoding", "/MacRomanEncoding")


def _characters(textpage):
    characters = []
    for index in range(textpage.count_chars()):
        text = textpage.get_text_range(index, 1)
        box = textpage.get_charbox(index)
        origin_x, origin_y = ctypes.c_double(), ctypes.c_double()
        raw.FPDFText_GetCharOrigin(textpage.raw, index, origin_x, origin_y)
        drawn = (box[2] - box[0]) > 0 or (box[3] - box[1]) > 0
        characters.append({
            "text": text,
            "box": box,
            "x": origin_x.value,
            "y": origin_y.value,
            "size": raw.FPDFText_GetFontSize(textpage.raw, index),
            "color": _color(textpage, index),
            "drawn": drawn,
        })
    return characters


def _color(textpage, index):
    channels = [ctypes.c_uint() for _ in range(4)]
    raw.FPDFText_GetFillColor(textpage.raw, index, *channels)
    return tuple(round(channel.value / 255, 4) for channel in channels[:3])


def _group(characters, visible):
    runs = []
    current = None
    for index in visible:
        character = characters[index]
        if not character["text"].strip() and not character["drawn"]:
            if current:
                current["pending"] = current.get("pending", "") + character["text"]
            continue
        if current and _continues(current, character):
            current["text"] += current.pop("pending", "") + character["text"]
            current["last"] = index
            current["box"] = _union(current["box"], character["box"])
        else:
            if current:
                runs.append(current)
            current = {
                "text": character["text"],
                "box": list(character["box"]),
                "size": character["size"],
                "color": character["color"],
                "x": character["x"],
                "y": character["y"],
                "first": index,
                "last": index,
            }
    if current:
        current.pop("pending", None)
        runs.append(current)
    for number, run in enumerate(runs):
        run["id"] = number
    return runs


def _continues(current, character):
    if abs(character["y"] - current["y"]) > LINE_TOLERANCE * max(current["size"], 1):
        return False
    return character["box"][0] - current["box"][2] <= GAP_TOLERANCE * max(current["size"], 1)


def _union(box, other):
    return [min(box[0], other[0]), min(box[1], other[1]), max(box[2], other[2]), max(box[3], other[3])]


def _mark_editable(runs, characters, contents, fonts):
    operations = show_operations(contents) if contents else []
    drawn = [index for index, character in enumerate(characters) if character["drawn"] or character["text"].strip()]
    codes = sum(len(operation["codes"]) for operation in operations)
    aligned = codes == len(drawn)

    owner = {}
    if aligned:
        position = 0
        for operation in operations:
            for offset in range(len(operation["codes"])):
                owner[drawn[position]] = (operation, offset)
                position += 1

    for run in runs:
        run["mode"] = "redraw"
        run["operation"] = None
        run["operations"] = None
        run["full_charset"] = False
        if not aligned:
            continue
        span = [index for index in range(run["first"], run["last"] + 1) if index in owner]
        if not span:
            continue

        used = []
        for index in span:
            operation = owner[index][0]
            if operation not in used:
                used.append(operation)
        whole = all(
            sum(1 for index in span if owner[index][0] is operation) == len(operation["codes"])
            for operation in used
        )
        if not whole:
            continue
        run["operations"] = used
        run["mode"] = "replaced"
        if len(used) != 1:
            continue

        operation = used[0]
        run["mode"] = "inplace"
        run["operation"] = operation
        run["encoding"] = {characters[index]["text"]: operation["codes"][owner[index][1]] for index in span}
        font = fonts.get(operation["font"])
        run["full_charset"] = is_standard(font.get_object() if font is not None else None)
    return runs
