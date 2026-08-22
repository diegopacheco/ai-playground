import pypdfium2

from png import encode

_documents = {}
_thumbnails = {}

THUMB_WIDTH = 260


def preview(sources, source_index, page_index, scale=1.5):
    page = _document(sources, source_index)[page_index]
    bitmap = page.render(scale=scale, rev_byteorder=True)
    return encode(bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.stride)


def thumbnail(sources, source_index, page_index):
    key = (source_index, page_index)
    if key not in _thumbnails:
        page = _document(sources, source_index)[page_index]
        bitmap = page.render(scale=THUMB_WIDTH / page.get_width(), rev_byteorder=True)
        _thumbnails[key] = encode(bytes(bitmap.buffer), bitmap.width, bitmap.height, bitmap.stride)
    return _thumbnails[key]


def forget(source_index):
    _documents.pop(source_index, None)
    for key in [key for key in _thumbnails if key[0] == source_index]:
        del _thumbnails[key]


def _document(sources, source_index):
    if source_index not in _documents:
        _documents[source_index] = pypdfium2.PdfDocument(sources[source_index])
    return _documents[source_index]
