import sys
import unittest

sys.path.insert(0, "src")

from pages import invert_pages, parse_pages


class ParsePages(unittest.TestCase):
    def test_order_is_the_users_order_so_extract_can_reorder(self):
        self.assertEqual(parse_pages("3,1", 5), [3, 1])

    def test_a_page_asked_for_twice_is_only_written_once(self):
        self.assertEqual(parse_pages("2,1-3", 5), [2, 1, 3])

    def test_range_is_inclusive_on_both_ends(self):
        self.assertEqual(parse_pages("2-4", 5), [2, 3, 4])

    def test_pages_are_one_based_so_page_zero_is_refused(self):
        with self.assertRaises(ValueError):
            parse_pages("0", 5)

    def test_a_page_past_the_end_fails_instead_of_being_dropped(self):
        with self.assertRaises(ValueError):
            parse_pages("6", 5)

    def test_backwards_range_fails_instead_of_selecting_nothing(self):
        with self.assertRaises(ValueError):
            parse_pages("4-2", 5)

    def test_non_numeric_input_fails_instead_of_being_ignored(self):
        with self.assertRaises(ValueError):
            parse_pages("first", 5)

    def test_empty_selection_fails_so_no_empty_pdf_is_written(self):
        with self.assertRaises(ValueError):
            parse_pages(" , ", 5)


class InvertPages(unittest.TestCase):
    def test_delete_keeps_the_rest_in_document_order(self):
        self.assertEqual(invert_pages([3, 1], 5), [2, 4, 5])

    def test_deleting_nothing_keeps_every_page(self):
        self.assertEqual(invert_pages([], 3), [1, 2, 3])


if __name__ == "__main__":
    unittest.main()
