import unittest
from typing import Any

from blograg.blogger import html_to_text, parse_entry


def entry(**overrides: Any) -> dict[str, Any]:
    base: dict[str, Any] = {
        "title": {"$t": " Burning Tokens "},
        "published": {"$t": "2026-09-27T21:05:51.369-03:00"},
        "content": {"$t": "<p>Tokens are <b>money</b>.</p>"},
        "category": [{"term": "AI"}, {"term": "culture"}],
        "link": [
            {"rel": "replies", "href": "https://blog/comments"},
            {"rel": "alternate", "href": "https://blog/2026/09/burning-tokens.html"},
        ],
    }
    return base | overrides


class HtmlToTextTest(unittest.TestCase):
    def test_markup_is_removed_so_only_prose_is_embedded(self) -> None:
        self.assertEqual(html_to_text("<p>Hello&nbsp;<b>world</b></p>\n<p>again</p>"), "Hello world again")

    def test_script_and_style_never_reach_the_index(self) -> None:
        html = "<style>p{color:red}</style><p>kept</p><script>track()</script>"
        self.assertEqual(html_to_text(html), "kept")


class ParseEntryTest(unittest.TestCase):
    def test_source_link_is_the_public_post_url_not_the_comments_feed(self) -> None:
        post = parse_entry(entry())
        assert post is not None
        self.assertEqual(post.url, "https://blog/2026/09/burning-tokens.html")

    def test_fields_used_for_filtering_are_extracted(self) -> None:
        post = parse_entry(entry())
        assert post is not None
        self.assertEqual(post.title, "Burning Tokens")
        self.assertEqual(post.labels, ("AI", "culture"))
        self.assertEqual(post.year, 2026)
        self.assertEqual(post.text, "Tokens are money .")

    def test_post_without_labels_is_still_indexed(self) -> None:
        post = parse_entry(entry(category=[]))
        assert post is not None
        self.assertEqual(post.labels, ())

    def test_entry_that_cannot_be_cited_or_has_no_text_is_skipped(self) -> None:
        self.assertIsNone(parse_entry(entry(link=[{"rel": "self", "href": "https://blog/feed"}])))
        self.assertIsNone(parse_entry(entry(content={"$t": "<p> </p>"})))


if __name__ == "__main__":
    unittest.main()
