import ctypes

import pypdfium2 as pdfium
import pypdfium2.raw as raw

from content import show_operations
from fontmap import reverse_map

LINE_TOLERANCE = 0.5
GAP_TOLERANCE = 0.9


def page_runs(page, contents, fonts=None, covers=()):
    textpage = page.get_textpage()
    characters = _characters(textpage)
    runs = _group(characters, range(len(characters)))
    _mark_editable(runs, characters, contents, fonts or {})
    return _uncovered(runs, covers)


def _uncovered(runs, covers):
    kept = []
    for run in runs:
        left, bottom, right, top = run["box"]
        buried = any(
            box[0] - 0.5 <= left and box[1] - 0.5 <= bottom
            and box[2] + 0.5 >= right and box[3] + 0.5 >= top
            and run["text"] != text
            for box, text in covers
        )
        if not buried:
            kept.append(run)
    for number, run in enumerate(kept):
        run["id"] = number
    return kept


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
            current["letters"] += character["text"]
            current["last"] = index
            current["box"] = _union(current["box"], character["box"])
        else:
            if current:
                runs.append(current)
            current = {
                "text": character["text"],
                "letters": character["text"],
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
    alphabets = {
        name: reverse_map(font.get_object() if hasattr(font, "get_object") else font)
        for name, font in fonts.items()
    }

    for run in runs:
        here = [operation for operation in operations if _inside(operation, run)]
        run["operations"] = here or None
        run["operation"] = None
        run["encoding"] = {}
        run["full_charset"] = False
        run["mode"] = "replaced" if here else "redraw"

        run["draw_size"] = here[0]["size"] * here[0]["scale"] if here else run["size"]
        names = {operation["font"] for operation in here}
        run["font_resource"] = names.pop() if len(names) == 1 else None
        _harvest(alphabets, here, run["letters"])

        if len(here) != 1:
            continue
        operation = here[0]
        if len(operation["codes"]) != len(run["letters"]):
            continue
        font = fonts.get(operation["font"])
        run["mode"] = "inplace"
        run["operation"] = operation
        run["encoding"] = dict(zip(
            run["letters"],
            [operation["codes"][at:at + 1] for at in range(len(operation["codes"]))],
        ))
        run["full_charset"] = is_standard(font.get_object() if font is not None else None)

    for run in runs:
        run["alphabet"] = alphabets.get(run["font_resource"], {})
        resource = fonts.get(run["font_resource"])
        run["standard_font"] = is_standard(
            resource.get_object() if hasattr(resource, "get_object") else resource
        )
    return runs


def _harvest(alphabets, operations, letters):
    if sum(len(operation["codes"]) for operation in operations) != len(letters):
        return
    position = 0
    for operation in operations:
        codes = operation["codes"]
        alphabets.setdefault(operation["font"], {}).update(
            zip(letters[position:position + len(codes)],
                [codes[at:at + 1] for at in range(len(codes))])
        )
        position += len(codes)


def _inside(operation, run):
    left, bottom, right, top = run["box"]
    slack = max(2.0, 0.35 * run["size"])
    if not left - slack <= operation["x"] <= right + slack:
        return False
    return abs(operation["y"] - run["y"]) <= 0.5 * max(run["size"], 1)
