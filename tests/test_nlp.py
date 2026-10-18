import unittest

import pandas as pd
from lyric_features import ROOT, load_metrics
from nlp import distinctive_words, fit_topics, load_stopwords, load_terms, match_play_counts


class MatchTests(unittest.TestCase):
    def setUp(self):
        self.metrics = pd.DataFrame({"song_id": ["a", "b", "c", "d"],
            "title": ["Playin' In The Band", "Sugar Magnolia", "Suite Part II   Grow", "Sugar Magnolia..."]})
        self.aliases = pd.DataFrame([{"setlist_title": "Grow", "lyrics_title": "Suite Part II   Grow", "note": ""}])

    def test_normalized_alias_instrumental_and_rank(self):
        plays = pd.DataFrame({"setlist_title": ["Drums", "Playin’ in the Band", "Sugar Magnolia", "Grow", "Unknown"],
            "times_played": [900, 600, 500, 400, 300]})
        out = match_play_counts(plays, self.metrics, self.aliases, top_n=2).set_index("setlist_title")
        self.assertNotIn("Drums", out.index)
        self.assertEqual(out.loc["Sugar Magnolia", "song_id"], "b")
        self.assertEqual(out.loc["Grow", "song_id"], "c")
        self.assertFalse(out.loc["Unknown", "matched"])
        self.assertEqual(out.top_50.sum(), 2)

    def test_two_setlist_rows_cannot_claim_one_text(self):
        aliases = pd.DataFrame([{"setlist_title": "Sunshine Daydream", "lyrics_title": "Sugar Magnolia", "note": ""}])
        plays = pd.DataFrame({"setlist_title": ["Sugar Magnolia", "Sunshine Daydream"], "times_played": [500, 100]})
        out = match_play_counts(plays, self.metrics, aliases)
        self.assertEqual(out.song_id.tolist(), ["b", ""])

    def test_ambiguous_title_raises(self):
        plays = pd.DataFrame({"setlist_title": ["sugar magnolia!"], "times_played": [1]})
        with self.assertRaises(ValueError):
            match_play_counts(plays, self.metrics, self.aliases)


class DistinctiveWordTests(unittest.TestCase):
    def test_direction_and_once_per_song(self):
        rows = [("g1", "train", 50), ("g2", "train", 1), ("g3", "train", 1)]
        rows += [(f"o{i}", "sorrow", 1) for i in range(6)] + [("o0", "train", 1)]
        rows += [(s, "river", 1) for s in ["g1", "g2", "g3", "o0", "o1", "o2"]]
        terms = pd.DataFrame(rows, columns=["song_id", "term", "count"])
        out = distinctive_words(terms, ["g1", "g2", "g3"], [f"o{i}" for i in range(6)], set(), min_songs=3).set_index("term")
        self.assertGreater(out.z["train"], 0)
        self.assertLess(out.z["sorrow"], 0)
        self.assertAlmostEqual(out.share_group["train"], 1.0)

    def test_overlapping_groups_rejected(self):
        terms = pd.DataFrame({"song_id": ["a"], "term": ["word"], "count": [1]})
        with self.assertRaises(ValueError):
            distinctive_words(terms, ["a"], ["a"], set())


class SnapshotTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.metrics = load_metrics()
        cls.terms = load_terms()

    def test_top_50_fully_matched(self):
        plays = match_play_counts(pd.read_csv(ROOT / "data/setlistfm_play_counts.csv"), self.metrics,
            pd.read_csv(ROOT / "data/title_aliases.csv"))
        top = plays[plays.top_50]
        self.assertEqual(len(top), 50)
        self.assertTrue(top.matched.all())
        self.assertEqual(top.iloc[0].setlist_title, "Me and My Uncle")

    def test_word_counts_reconcile(self):
        counts = self.terms.groupby("song_id")["count"].sum().sort_index()
        pd.testing.assert_series_equal(counts, self.metrics.set_index("song_id").word_count.sort_index(),
            check_names=False)

    def test_topics_are_reproducible(self):
        ids = self.metrics.song_id.tolist()
        first, songs = fit_topics(self.terms, ids, load_stopwords(), k=7)
        second, _ = fit_topics(self.terms, ids, load_stopwords(), k=7)
        pd.testing.assert_frame_equal(first, second)
        shares = songs.filter(like="topic_").sum(axis=1)
        self.assertTrue(((shares - 1).abs() < 1e-9).all())


if __name__ == "__main__":
    unittest.main()
