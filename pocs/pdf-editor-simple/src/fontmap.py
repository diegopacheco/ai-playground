from content import tokenize


def reverse_map(font):
    if font is None:
        return {}
    stream = font.get("/ToUnicode")
    if stream is None:
        return {}
    try:
        data = stream.get_object().get_data()
    except AttributeError:
        return {}
    return _parse(data)


def _parse(data):
    tokens = tokenize(data)
    mapping = {}
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.kind == "operator" and token.value == "beginbfchar":
            index = _read_chars(tokens, index + 1, mapping)
        elif token.kind == "operator" and token.value == "beginbfrange":
            index = _read_ranges(tokens, index + 1, mapping)
        else:
            index += 1
    return mapping


def _read_chars(tokens, index, mapping):
    while index + 1 < len(tokens) and tokens[index].kind == "string":
        _add(mapping, tokens[index].value, tokens[index + 1].value)
        index += 2
    return index


def _read_ranges(tokens, index, mapping):
    while index + 2 < len(tokens) and tokens[index].kind == "string":
        low, high = tokens[index].value, tokens[index + 1].value
        following = tokens[index + 2]
        if following.kind == "[":
            index += 3
            code = int.from_bytes(low, "big")
            while index < len(tokens) and tokens[index].kind == "string":
                _add(mapping, code.to_bytes(len(low), "big"), tokens[index].value)
                code += 1
                index += 1
            index += 1
        else:
            _add_range(mapping, low, high, following.value)
            index += 3
    return index


def _add_range(mapping, low, high, destination):
    first = int.from_bytes(low, "big")
    last = int.from_bytes(high, "big")
    if last - first > 65535:
        return
    start = int.from_bytes(destination, "big")
    width = len(destination)
    for offset in range(last - first + 1):
        _add(mapping, (first + offset).to_bytes(len(low), "big"), (start + offset).to_bytes(width, "big"))


def _add(mapping, code, destination):
    try:
        text = destination.decode("utf-16-be")
    except (UnicodeDecodeError, ValueError):
        return
    if len(text) == 1 and text not in mapping:
        mapping[text] = code
