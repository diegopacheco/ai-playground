DELIMITERS = b"()<>[]{}/%"
WHITESPACE = b"\x00\t\n\x0c\r "

ESCAPES = {b"n": b"\n", b"r": b"\r", b"t": b"\t", b"b": b"\b", b"f": b"\x0c",
           b"(": b"(", b")": b")", b"\\": b"\\"}


class Token:
    def __init__(self, kind, value, start, end):
        self.kind = kind
        self.value = value
        self.start = start
        self.end = end

    def __repr__(self):
        return f"Token({self.kind}, {self.value!r})"


def tokenize(data):
    tokens = []
    position = 0
    size = len(data)
    while position < size:
        byte = data[position:position + 1]
        if byte in WHITESPACE:
            position += 1
        elif byte == b"%":
            while position < size and data[position:position + 1] not in b"\r\n":
                position += 1
        elif byte == b"(":
            value, position = _literal_string(data, position)
            tokens.append(Token("string", value, tokens and position or position, position))
            tokens[-1].start = value[1]
            tokens[-1].end = position
            tokens[-1].value = value[0]
        elif byte == b"<" and data[position + 1:position + 2] != b"<":
            end = data.index(b">", position) + 1
            digits = b"".join(data[position + 1:end - 1].split())
            if len(digits) % 2:
                digits += b"0"
            tokens.append(Token("string", bytes.fromhex(digits.decode("latin-1")), position, end))
            position = end
        elif byte == b"/":
            end = position + 1
            while end < size and data[end:end + 1] not in WHITESPACE and data[end:end + 1] not in DELIMITERS:
                end += 1
            tokens.append(Token("name", data[position:end].decode("latin-1"), position, end))
            position = end
        elif byte in b"[]<>{}":
            tokens.append(Token(byte.decode("latin-1"), byte.decode("latin-1"), position, position + 1))
            position += 1 if byte not in b"<>" else 2
        else:
            end = position
            while end < size and data[end:end + 1] not in WHITESPACE and data[end:end + 1] not in DELIMITERS:
                end += 1
            if end == position:
                end += 1
            raw = data[position:end]
            kind = "number" if _is_number(raw) else "operator"
            value = float(raw) if kind == "number" else raw.decode("latin-1")
            tokens.append(Token(kind, value, position, end))
            position = end
    return tokens


def _literal_string(data, position):
    start = position
    position += 1
    depth = 1
    out = bytearray()
    while position < len(data):
        byte = data[position:position + 1]
        if byte == b"\\":
            following = data[position + 1:position + 2]
            if following in ESCAPES:
                out += ESCAPES[following]
                position += 2
            elif following.isdigit():
                digits = data[position + 1:position + 4]
                keep = 0
                while keep < 3 and digits[keep:keep + 1].isdigit():
                    keep += 1
                out.append(int(digits[:keep], 8) & 0xFF)
                position += 1 + keep
            else:
                position += 2
        elif byte == b"(":
            depth += 1
            out += byte
            position += 1
        elif byte == b")":
            depth -= 1
            position += 1
            if depth == 0:
                break
            out += byte
        else:
            out += byte
            position += 1
    return (bytes(out), start), position


def _is_number(raw):
    try:
        float(raw)
        return True
    except ValueError:
        return False


SHOW = {"Tj", "TJ", "'", '"'}
IDENTITY = (1.0, 0.0, 0.0, 1.0, 0.0, 0.0)


def multiply(first, second):
    a1, b1, c1, d1, e1, f1 = first
    a2, b2, c2, d2, e2, f2 = second
    return (
        a1 * a2 + b1 * c2,
        a1 * b2 + b1 * d2,
        c1 * a2 + d1 * c2,
        c1 * b2 + d1 * d2,
        e1 * a2 + f1 * c2 + e2,
        e1 * b2 + f1 * d2 + f2,
    )


def show_operations(data):
    tokens = tokenize(data)
    operands = []
    font = None
    size = 0.0
    ctm = IDENTITY
    stack = []
    matrix = IDENTITY
    line_matrix = IDENTITY
    leading = 0.0
    operations = []

    def numbers(count):
        values = [item.value for item in operands if item.kind == "number"]
        return values[-count:] if len(values) >= count else None

    for token in tokens:
        if token.kind == "operator":
            name = token.value
            if name == "q":
                stack.append(ctm)
            elif name == "Q" and stack:
                ctm = stack.pop()
            elif name == "cm" and numbers(6):
                ctm = multiply(tuple(numbers(6)), ctm)
            elif name == "BT":
                matrix = line_matrix = IDENTITY
            elif name == "Tf" and len(operands) >= 2:
                font, size = operands[-2].value, operands[-1].value
            elif name == "TL" and numbers(1):
                leading = numbers(1)[0]
            elif name == "Tm" and numbers(6):
                matrix = line_matrix = tuple(numbers(6))
            elif name in ("Td", "TD") and numbers(2):
                shift_x, shift_y = numbers(2)
                if name == "TD":
                    leading = -shift_y
                line_matrix = multiply((1.0, 0.0, 0.0, 1.0, shift_x, shift_y), line_matrix)
                matrix = line_matrix
            elif name == "T*":
                line_matrix = multiply((1.0, 0.0, 0.0, 1.0, 0.0, -leading), line_matrix)
                matrix = line_matrix

            if name in SHOW:
                if name in ("'", '"'):
                    line_matrix = multiply((1.0, 0.0, 0.0, 1.0, 0.0, -leading), line_matrix)
                    matrix = line_matrix
                strings = [item for item in operands if item.kind == "string"]
                if strings:
                    placed = multiply(matrix, ctm)
                    operations.append({
                        "operator": name,
                        "font": font,
                        "size": size,
                        "codes": b"".join(item.value for item in strings),
                        "start": operands[0].start,
                        "end": token.end,
                        "single": name == "Tj" and len(operands) == 1,
                        "x": placed[4],
                        "y": placed[5],
                    })
            operands = []
        elif token.kind in ("[", "]", "<<", ">>"):
            continue
        else:
            operands.append(token)
    return operations
