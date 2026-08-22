from io import BytesIO

from pypdf import PdfReader, PdfWriter

import render


class Document:
    def __init__(self):
        self.reset()

    def reset(self):
        self.name = ""
        self.sources = []
        self.slots = []
        self.history = []
        self._next_uid = 1

    def open(self, name, data):
        for index in range(len(self.sources)):
            render.forget(index)
        self.reset()
        self.name = name
        self.add(data)
        self.history.clear()

    def add(self, data):
        self._remember()
        source_index = len(self.sources)
        self.sources.append(data)
        for page_index in range(len(PdfReader(BytesIO(data)).pages)):
            self.slots.append({"uid": self._take_uid(), "source": source_index, "page": page_index, "rotation": 0})

    def delete(self, uids):
        chosen = self._chosen(uids)
        if len(chosen) == len(self.slots):
            raise ValueError("deleting every page would leave an empty document")
        self._remember()
        self.slots = [slot for slot in self.slots if slot["uid"] not in chosen]

    def keep(self, uids):
        chosen = self._chosen(uids)
        self._remember()
        self.slots = [slot for slot in self.slots if slot["uid"] in chosen]

    def rotate(self, uids, angle):
        if angle % 90:
            raise ValueError("angle must be a multiple of 90")
        chosen = self._chosen(uids)
        self._remember()
        for slot in self.slots:
            if slot["uid"] in chosen:
                slot["rotation"] = (slot["rotation"] + angle) % 360

    def reorder(self, uids):
        if sorted(uids) != sorted(slot["uid"] for slot in self.slots):
            raise ValueError("the new order must list every page exactly once")
        self._remember()
        by_uid = {slot["uid"]: slot for slot in self.slots}
        self.slots = [by_uid[uid] for uid in uids]

    def undo(self):
        if not self.history:
            raise ValueError("nothing to undo")
        self.slots = self.history.pop()

    def save(self):
        writer = PdfWriter()
        readers = {}
        for slot in self.slots:
            source_index = slot["source"]
            if source_index not in readers:
                readers[source_index] = PdfReader(BytesIO(self.sources[source_index]))
            page = readers[source_index].pages[slot["page"]]
            if slot["rotation"]:
                page.rotate(slot["rotation"])
            writer.add_page(page)
        buffer = BytesIO()
        writer.write(buffer)
        return buffer.getvalue()

    def state(self):
        return {
            "name": self.name,
            "canUndo": bool(self.history),
            "pages": [
                {
                    "uid": slot["uid"],
                    "rotation": slot["rotation"],
                    "thumb": f"/thumb/{slot['source']}/{slot['page']}.png",
                }
                for slot in self.slots
            ],
        }

    def thumbnail(self, source_index, page_index):
        return render.thumbnail(self.sources, source_index, page_index)

    def _chosen(self, uids):
        chosen = set(uids)
        known = {slot["uid"] for slot in self.slots}
        if not chosen:
            raise ValueError("no pages selected")
        if not chosen <= known:
            raise ValueError("selection refers to pages that are gone")
        return chosen

    def _remember(self):
        self.history.append([dict(slot) for slot in self.slots])
        del self.history[:-30]

    def _take_uid(self):
        uid = self._next_uid
        self._next_uid += 1
        return uid
