import sys
import unittest
from io import BytesIO

sys.path.insert(0, "src")

from pypdf import PdfReader

from content import show_operations, tokenize
from sample import assemble, build
from textedit import apply_edits, covered_boxes, read_runs


def _text(data, page=0):
    return PdfReader(BytesIO(data)).pages[page].extract_text().strip()


class Tokenizer(unittest.TestCase):
    def test_escaped_parenthesis_does_not_end_the_string(self):
        tokens = tokenize(rb"(a\(b\)c) Tj")
        self.assertEqual(tokens[0].value, b"a(b)c")

    def test_octal_escape_becomes_one_byte(self):
        self.assertEqual(tokenize(rb"(\101) Tj")[0].value, b"A")

    def test_hex_string_is_decoded(self):
        self.assertEqual(tokenize(b"<4142> Tj")[0].value, b"AB")

    def test_comments_are_ignored_so_markers_are_not_operators(self):
        self.assertEqual([t.value for t in tokenize(b"% a comment\n12 Tf")], [12.0, "Tf"])

    def test_kerned_array_counts_as_one_show_operation(self):
        operations = show_operations(b"BT /F1 12 Tf [(He)-30(llo)]TJ ET")
        self.assertEqual(len(operations), 1)
        self.assertEqual(operations[0]["codes"], b"Hello")
        self.assertEqual(operations[0]["font"], "/F1")
        self.assertEqual(operations[0]["size"], 12.0)


class InPlace(unittest.TestCase):
    def setUp(self):
        self.pdf = build(["Page 1"])

    def test_a_single_show_operation_in_a_standard_font_is_edited_in_place(self):
        width, height, runs = read_runs(self.pdf, 0)
        self.assertEqual(runs[0]["mode"], "inplace")

    def test_editing_in_place_replaces_the_text_and_nothing_else(self):
        data, report = apply_edits(self.pdf, 0, {0: "Replaced"})
        self.assertEqual(report, [{"run": 0, "mode": "inplace"}])
        self.assertEqual(_text(data), "Replaced")

    def test_in_place_edit_leaves_no_covering_rectangle_behind(self):
        data, _ = apply_edits(self.pdf, 0, {0: "Replaced"})
        contents = PdfReader(BytesIO(data)).pages[0].get_contents().get_data()
        self.assertEqual(covered_boxes(contents), [])

    def test_the_edited_line_can_be_edited_again(self):
        data, _ = apply_edits(self.pdf, 0, {0: "First"})
        data, report = apply_edits(data, 0, {0: "Second"})
        self.assertEqual(report[0]["mode"], "inplace")
        self.assertEqual(_text(data), "Second")

    def test_a_character_no_font_here_can_draw_is_refused_instead_of_mangled(self):
        with self.assertRaises(ValueError) as refused:
            apply_edits(_split_line_pdf(), 0, {0: "漢字"})
        self.assertIn("cannot draw", str(refused.exception))

    def test_an_unknown_line_number_is_refused(self):
        with self.assertRaises(ValueError):
            apply_edits(self.pdf, 0, {99: "nowhere"})


def _split_line_pdf():
    stream = "BT /F1 24 Tf 72 700 Td (Hello ) Tj (world) Tj ET"
    return assemble({
        1: "<< /Type /Catalog /Pages 2 0 R >>",
        2: "<< /Type /Pages /Kids [4 0 R] /Count 1 >>",
        3: "<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>",
        4: ("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
            "/Resources << /Font << /F1 3 0 R >> >> /Contents 5 0 R >>"),
        5: f"<< /Length {len(stream)} >>\nstream\n{stream}\nendstream",
    })


class Redraw(unittest.TestCase):
    def setUp(self):
        self.pdf = _split_line_pdf()
        self.data, self.report = apply_edits(self.pdf, 0, {0: "Hello again"})

    def test_a_line_split_across_two_operations_cannot_be_edited_in_place(self):
        _, _, runs = read_runs(self.pdf, 0)
        self.assertEqual(runs[0]["text"], "Hello world")
        self.assertEqual(runs[0]["mode"], "replaced")
        self.assertEqual(self.report, [{"run": 0, "mode": "replaced"}])

    def test_the_old_text_is_removed_rather_than_hidden_under_a_rectangle(self):
        contents = PdfReader(BytesIO(self.data)).pages[0].get_contents().get_data()
        self.assertEqual(covered_boxes(contents), [])
        self.assertNotIn(b"(Hello ) Tj", contents)

    def test_the_replaced_line_leaves_the_page_text_clean(self):
        self.assertEqual(_text(self.data), "Hello again")

    def test_the_covered_original_is_hidden_from_the_editor(self):
        _, _, runs = read_runs(self.data, 0)
        self.assertNotIn("Hello world", [run["text"] for run in runs])

    def test_the_redrawn_text_stays_visible_in_the_editor(self):
        _, _, runs = read_runs(self.data, 0)
        self.assertIn("Hello again", [run["text"] for run in runs])

    def test_the_redrawn_line_keeps_the_position_of_the_line_it_replaced(self):
        _, _, before = read_runs(self.pdf, 0)
        _, _, after = read_runs(self.data, 0)
        replacement = [run for run in after if run["text"] == "Hello again"][0]
        self.assertAlmostEqual(before[0]["box"][0], replacement["box"][0], delta=2.0)


class Grouping(unittest.TestCase):
    def test_one_printed_line_is_offered_as_one_editable_run(self):
        width, height, runs = read_runs(build(["A single line of text"]), 0)
        self.assertEqual([run["text"] for run in runs], ["A single line of text"])

    def test_page_size_is_reported_in_points(self):
        width, height, _ = read_runs(build(["Page 1"]), 0)
        self.assertEqual((round(width), round(height)), (612, 792))


if __name__ == "__main__":
    unittest.main()
