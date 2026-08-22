import sys
import unittest
from io import BytesIO

sys.path.insert(0, "src")

from pypdf import PdfReader

from document import Document
from sample import build


def _pdf(count, prefix="Page"):
    return build([f"{prefix} {number}" for number in range(1, count + 1)])


def _titles(data):
    return [page.extract_text().strip() for page in PdfReader(BytesIO(data)).pages]


class Editing(unittest.TestCase):
    def setUp(self):
        self.document = Document()
        self.document.open("deck.pdf", _pdf(4))
        self.uids = [page["uid"] for page in self.document.state()["pages"]]

    def test_saved_file_follows_the_order_shown_on_screen(self):
        self.document.reorder(list(reversed(self.uids)))
        self.assertEqual(_titles(self.document.save()), ["Page 4", "Page 3", "Page 2", "Page 1"])

    def test_rotation_reaches_the_saved_file(self):
        self.document.rotate([self.uids[1]], 90)
        rotations = [page.rotation for page in PdfReader(BytesIO(self.document.save())).pages]
        self.assertEqual(rotations, [0, 90, 0, 0])

    def test_rotating_twice_accumulates_and_wraps_at_360(self):
        self.document.rotate([self.uids[0]], 270)
        self.document.rotate([self.uids[0]], 180)
        self.assertEqual(self.document.state()["pages"][0]["rotation"], 90)

    def test_delete_removes_only_the_selected_pages(self):
        self.document.delete([self.uids[0], self.uids[2]])
        self.assertEqual(_titles(self.document.save()), ["Page 2", "Page 4"])

    def test_keep_only_drops_everything_else(self):
        self.document.keep([self.uids[2]])
        self.assertEqual(_titles(self.document.save()), ["Page 3"])

    def test_deleting_every_page_is_refused_so_the_editor_is_never_empty(self):
        with self.assertRaises(ValueError):
            self.document.delete(self.uids)

    def test_reorder_must_account_for_every_page_so_none_is_lost(self):
        with self.assertRaises(ValueError):
            self.document.reorder(self.uids[:2])

    def test_added_pdf_keeps_its_own_pages_and_appends_them(self):
        self.document.add(_pdf(2, "Extra"))
        self.assertEqual(_titles(self.document.save()), ["Page 1", "Page 2", "Page 3", "Page 4", "Extra 1", "Extra 2"])

    def test_undo_restores_the_pages_that_were_deleted(self):
        self.document.delete([self.uids[0]])
        self.document.undo()
        self.assertEqual(_titles(self.document.save()), ["Page 1", "Page 2", "Page 3", "Page 4"])

    def test_undo_after_reorder_restores_the_previous_order(self):
        self.document.reorder(list(reversed(self.uids)))
        self.document.undo()
        self.assertEqual([page["uid"] for page in self.document.state()["pages"]], self.uids)

    def test_a_freshly_opened_file_has_nothing_to_undo(self):
        self.assertFalse(self.document.state()["canUndo"])

    def test_undo_with_no_history_fails_instead_of_doing_nothing(self):
        with self.assertRaises(ValueError):
            self.document.undo()

    def test_opening_a_second_file_replaces_the_first(self):
        self.document.open("other.pdf", _pdf(2, "Other"))
        self.assertEqual(_titles(self.document.save()), ["Other 1", "Other 2"])

    def test_selection_that_names_a_deleted_page_is_refused(self):
        self.document.delete([self.uids[0]])
        with self.assertRaises(ValueError):
            self.document.rotate([self.uids[0]], 90)

    def test_rotation_must_be_a_quarter_turn(self):
        with self.assertRaises(ValueError):
            self.document.rotate([self.uids[0]], 45)


if __name__ == "__main__":
    unittest.main()


class Annotations(unittest.TestCase):
    def setUp(self):
        self.document = Document()
        self.document.open("deck.pdf", _pdf(2))
        self.uid = self.document.state()["pages"][0]["uid"]
        self.marker = {"kind": "highlight", "x": 72, "y": 690, "width": 120, "height": 30,
                       "text": "", "size": 12, "color": [1, 0.92, 0.23]}
        self.written = {"kind": "text", "x": 72, "y": 600, "width": 200, "height": 20,
                        "text": "Reviewed", "size": 14, "color": [0.85, 0.1, 0.1]}

    def test_an_annotation_shows_up_on_the_page_it_was_added_to(self):
        self.document.add_note(self.uid, self.marker)
        pages = self.document.state()["pages"]
        self.assertEqual(len(pages[0]["notes"]), 1)
        self.assertEqual(pages[1]["notes"], [])

    def test_moving_an_annotation_keeps_it_a_single_annotation(self):
        note_id = self.document.add_note(self.uid, self.marker)
        self.document.update_note(self.uid, note_id, {"x": 200, "y": 400})
        notes = self.document.state()["pages"][0]["notes"]
        self.assertEqual(len(notes), 1)
        self.assertEqual((notes[0]["x"], notes[0]["y"]), (200, 400))

    def test_a_deleted_annotation_is_gone(self):
        note_id = self.document.add_note(self.uid, self.marker)
        self.document.delete_note(self.uid, note_id)
        self.assertEqual(self.document.state()["pages"][0]["notes"], [])

    def test_undo_brings_a_deleted_annotation_back(self):
        self.document.add_note(self.uid, self.marker)
        self.document.delete_note(self.uid, self.marker and 1)
        self.document.undo()
        self.assertEqual(len(self.document.state()["pages"][0]["notes"]), 1)

    def test_written_text_reaches_the_saved_file(self):
        self.document.add_note(self.uid, self.written)
        self.assertIn("Reviewed", _titles(self.document.save())[0])

    def test_a_highlight_is_drawn_as_a_filled_rectangle_in_the_saved_file(self):
        self.document.add_note(self.uid, self.marker)
        contents = PdfReader(BytesIO(self.document.save())).pages[0].get_contents().get_data()
        self.assertIn(b" re f", contents)

    def test_annotations_do_not_touch_the_page_they_are_not_on(self):
        self.document.add_note(self.uid, self.written)
        self.assertNotIn("Reviewed", _titles(self.document.save())[1])

    def test_the_page_keeps_its_own_text_under_an_annotation(self):
        self.document.add_note(self.uid, self.written)
        self.assertIn("Page 1", _titles(self.document.save())[0])
