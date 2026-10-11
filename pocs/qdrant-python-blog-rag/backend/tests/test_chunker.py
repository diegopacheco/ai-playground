import unittest

from blograg.chunker import chunk_words


def words(count: int) -> str:
    return " ".join(f"w{number}" for number in range(count))


class ChunkerTest(unittest.TestCase):
    def test_short_post_stays_whole_so_small_posts_are_still_searchable(self) -> None:
        self.assertEqual(chunk_words(words(30)), [words(30)])

    def test_empty_post_produces_nothing_to_embed(self) -> None:
        self.assertEqual(chunk_words("   "), [])

    def test_no_word_is_lost_when_a_post_is_split(self) -> None:
        chunks = chunk_words(words(1000), size=220, overlap=40)
        seen = {word for chunk in chunks for word in chunk.split()}
        self.assertEqual(seen, set(words(1000).split()))

    def test_neighbour_chunks_share_the_overlap_so_ideas_cut_at_a_boundary_survive(self) -> None:
        first, second = chunk_words(words(300), size=220, overlap=40)[:2]
        self.assertEqual(first.split()[-40:], second.split()[:40])

    def test_tail_already_covered_by_the_overlap_is_not_duplicated(self) -> None:
        self.assertEqual(len(chunk_words(words(220), size=220, overlap=40)), 1)
        self.assertEqual(len(chunk_words(words(221), size=220, overlap=40)), 2)


if __name__ == "__main__":
    unittest.main()
