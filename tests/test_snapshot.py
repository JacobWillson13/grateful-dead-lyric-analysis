import json
import unittest

from lyric_features import ROOT, load_metrics


class SnapshotTests(unittest.TestCase):
    def test_manifest_and_numerical_snapshot_agree(self):
        manifest = json.loads((ROOT/"data/source_manifest.json").read_text())
        frame = load_metrics()
        successful = {x["song_id"]: x for x in manifest["sources"] if x["status"] == "ok"}
        self.assertEqual(set(frame.song_id), set(successful))
        self.assertTrue(all(row.raw_sha256 == successful[row.song_id]["raw_sha256"] for row in frame.itertuples()))
        self.assertFalse(any("lyric" == str(c).lower() or "lyrics" == str(c).lower() for c in frame.columns))
        self.assertTrue(frame.loc[frame.word_count.lt(50), "mattr_50"].isna().all())
        self.assertTrue(frame.loc[frame.word_count.ge(50), "mattr_50"].between(0,1).all())


if __name__ == "__main__":
    unittest.main()
