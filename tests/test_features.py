import math
import tempfile
import unittest
from pathlib import Path

import pandas as pd
from lyric_features import clean_source, load_metrics, mattr, text_metrics, title_key, tokenize


class FeatureTests(unittest.TestCase):
    def test_unicode_contractions_and_numbers(self):
        self.assertEqual(tokenize("Don't don’t 123 naïve blue-green"), ["don't", "don't", "naïve", "blue", "green"])

    def test_fixed_window_example(self):
        self.assertAlmostEqual(mattr(["alpha", "alpha", "beta"], 2), .75)
        self.assertTrue(math.isnan(mattr(["alpha"], 2)))
        with self.assertRaises(ValueError):
            mattr(["alpha"], 0)

    def test_length_confounding_demonstration(self):
        short = "alpha beta " * 25
        long = short * 4
        a, b = text_metrics(short), text_metrics(long)
        self.assertGreater(a["type_token_ratio"], b["type_token_ratio"])
        self.assertAlmostEqual(a["mattr_50"], b["mattr_50"])

    def test_empty_text_and_exact_repetition(self):
        self.assertEqual(text_metrics("")["word_count"], 0)
        self.assertTrue(math.isnan(text_metrics("")["type_token_ratio"]))
        result = text_metrics("Alpha beta!\nGamma delta\nALPHA, BETA")
        self.assertAlmostEqual(result["repeated_line_share"], 1/3)

    def test_header_removal_is_exact_not_substring(self):
        cleaned, removed, flags = clean_source("\nExample Title\n\nalpha beta", "example-title")
        self.assertTrue(removed)
        self.assertEqual(cleaned, "alpha beta")
        self.assertEqual(flags, 0)
        kept, removed, _ = clean_source("Example Title continues\nalpha", "Example Title")
        self.assertFalse(removed)
        self.assertTrue(kept.startswith("Example Title continues"))

    def test_annotation_flag_preserves_text(self):
        text, _, flags = clean_source("Example\n[editor note]\nalpha beta", "Example")
        self.assertEqual(flags, 1)
        self.assertIn("[editor note]", text)

    def test_title_normalization(self):
        self.assertEqual(title_key("Playin’ in the Band"), title_key("Playin' In The Band"))

    def test_derived_schema_checked(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/"bad.csv"
            pd.DataFrame({"other": [1]}).to_csv(path, index=False)
            with self.assertRaises(ValueError):
                load_metrics(path)


if __name__ == "__main__":
    unittest.main()
