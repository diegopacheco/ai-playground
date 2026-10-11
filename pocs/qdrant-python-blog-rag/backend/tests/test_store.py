import unittest
import uuid

from qdrant_client import models

from blograg.store import build_filter, point_id, to_hit


class PointIdTest(unittest.TestCase):
    def test_same_chunk_gets_the_same_id_so_reindexing_overwrites_instead_of_duplicating(self) -> None:
        self.assertEqual(point_id("https://blog/a.html", 2), point_id("https://blog/a.html", 2))

    def test_chunks_of_one_post_and_chunks_of_other_posts_never_collide(self) -> None:
        ids = {point_id("https://blog/a.html", 0), point_id("https://blog/a.html", 1), point_id("https://blog/b.html", 0)}
        self.assertEqual(len(ids), 3)

    def test_id_is_a_uuid_because_qdrant_only_accepts_uuids_or_integers(self) -> None:
        self.assertEqual(str(uuid.UUID(point_id("https://blog/a.html", 0))), point_id("https://blog/a.html", 0))


class BuildFilterTest(unittest.TestCase):
    def test_no_filter_searches_the_whole_blog(self) -> None:
        self.assertIsNone(build_filter(None, None))
        self.assertIsNone(build_filter("", 0))

    def test_label_and_year_must_both_match(self) -> None:
        built = build_filter("AI", 2026)
        assert built is not None and isinstance(built.must, list)
        conditions = [condition for condition in built.must if isinstance(condition, models.FieldCondition)]
        matched = {condition.key: condition.match for condition in conditions}
        self.assertEqual(matched, {"labels": models.MatchValue(value="AI"), "year": models.MatchValue(value=2026)})

    def test_single_filter_does_not_constrain_the_other_field(self) -> None:
        built = build_filter(None, 2024)
        assert built is not None and isinstance(built.must, list)
        self.assertEqual(len(built.must), 1)


class ToHitTest(unittest.TestCase):
    def test_hit_carries_what_the_ui_needs_to_cite_the_post(self) -> None:
        payload = {"url": "u", "title": "t", "published": "2026-01-01", "labels": ["AI"], "text": "body", "chunk": 3}
        hit = to_hit(payload, 0.87654321)
        self.assertEqual((hit.url, hit.title, hit.labels, hit.text, hit.score), ("u", "t", ["AI"], "body", 0.8765))


if __name__ == "__main__":
    unittest.main()
