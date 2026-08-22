from io import BytesIO

from pypdf import PdfReader, PdfWriter

import render
from textedit import apply_changes, bake, page_shape, read_runs, to_page


class Document:
    def __init__(self):
        self.reset()

    def reset(self):
        self.name = ""
        self.sources = []
        self.versions = []
        self.notes = {}
        self._next_note = 1
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
        self.versions.append(0)
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
        previous = self.history.pop()
        self.slots = previous["slots"]
        self.notes = previous["notes"]
        for index, data in enumerate(previous["sources"]):
            if index < len(self.sources) and self.sources[index] is not data:
                self.sources[index] = data
                self.versions[index] += 1
                render.forget(index)

    def runs(self, uid):
        slot = self._slot(uid)
        return read_runs(self.sources[slot["source"]], slot["page"])

    def move_text(self, uid, run_id, dx, dy):
        return self._change(uid, {run_id: {"text": None, "dx": dx, "dy": dy}})

    def edit_text(self, uid, edits):
        if not edits:
            raise ValueError("no text was changed")
        return self._change(uid, {
            run_id: {"text": text, "dx": 0.0, "dy": 0.0} for run_id, text in edits.items()
        })

    def add_note(self, uid, note):
        slot = self._slot(uid)
        self._remember()
        note = dict(note, id=self._next_note)
        self._next_note += 1
        self.notes.setdefault(uid, []).append(self._to_page(slot, note))
        return note["id"]

    def update_note(self, uid, note_id, changes):
        slot = self._slot(uid)
        note = self._note(uid, note_id)
        self._remember()
        note.update(self._to_page(slot, {**note, **changes}))

    def delete_note(self, uid, note_id):
        self._note(uid, note_id)
        self._remember()
        self.notes[uid] = [note for note in self.notes[uid] if note["id"] != note_id]

    def _note(self, uid, note_id):
        for note in self.notes.get(uid, []):
            if note["id"] == note_id:
                return note
        raise ValueError(f"there is no annotation {note_id} on this page")

    def _to_page(self, slot, note):
        rotation, width, height = page_shape(self.sources[slot["source"]], slot["page"])
        if not rotation:
            return note
        left, bottom = to_page((note["x"], note["y"]), rotation, width, height)
        right, top = to_page((note["x"] + note["width"], note["y"] + note["height"]),
                             rotation, width, height)
        return {**note, "x": min(left, right), "y": min(bottom, top),
                "width": abs(right - left), "height": abs(top - bottom)}

    def _change(self, uid, changes):
        slot = self._slot(uid)
        self._remember()
        data, report = apply_changes(self.sources[slot["source"]], slot["page"], changes)
        self.sources[slot["source"]] = data
        self.versions[slot["source"]] += 1
        render.forget(slot["source"])
        return report

    def _displayed(self, slot, note):
        return self._to_page(slot, note)

    def _slot(self, uid):
        for slot in self.slots:
            if slot["uid"] == uid:
                return slot
        raise ValueError(f"page {uid} is not in the document")

    def save(self):
        writer = PdfWriter()
        sources = list(self.sources)
        for slot in self.slots:
            notes = self.notes.get(slot["uid"])
            if notes:
                sources[slot["source"]] = bake(sources[slot["source"]], slot["page"], notes)
        readers = {}
        for slot in self.slots:
            source_index = slot["source"]
            if source_index not in readers:
                readers[source_index] = PdfReader(BytesIO(sources[source_index]))
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
                    "thumb": f"/thumb/{slot['source']}/{slot['page']}.png?v={self.versions[slot['source']]}",
                    "view": f"/view/{slot['source']}/{slot['page']}.png?v={self.versions[slot['source']]}",
                    "notes": [self._displayed(slot, note) for note in self.notes.get(slot["uid"], [])],
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
        self.history.append({
            "slots": [dict(slot) for slot in self.slots],
            "sources": list(self.sources),
            "notes": {uid: [dict(note) for note in notes] for uid, notes in self.notes.items()},
        })
        del self.history[:-30]

    def _take_uid(self):
        uid = self._next_uid
        self._next_uid += 1
        return uid
