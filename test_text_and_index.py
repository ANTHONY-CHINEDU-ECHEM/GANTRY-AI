import time
import unittest

import numpy as np

from gantry_ai.text import stem, tokenize, with_bigrams
from gantry_ai.utils import sub, top_k_indices, wilson_lower
from tests.helpers import engine


class TextTests(unittest.TestCase):
    def test_tokenize_removes_stopwords_and_stems(self):
        self.assertEqual(tokenize("The suppliers are missing deliveries"), ["supplier", "miss", "delivery"])

    def test_stem_is_conservative(self):
        self.assertEqual(stem("cost"), "cost")
        self.assertEqual(stem("resigning"), stem("resigned"))

    def test_bigrams(self):
        self.assertEqual(with_bigrams(["a", "b", "c"]), ["a", "b", "c", "a_b", "b_c"])


class UtilityTests(unittest.TestCase):
    def test_top_k_matches_full_sort(self):
        rng = np.random.default_rng(0)
        scores = rng.random(5000)
        self.assertEqual(list(top_k_indices(scores, 20)), list(np.flip(np.argsort(scores))[:20]))

    def test_wilson_prefers_more_evidence(self):
        small, large = wilson_lower([9, 160], [10, 200])
        self.assertGreater(large, small)
        self.assertEqual(float(wilson_lower(0, 0)), 0.0)


class IndexTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_bm25_and_dense_find_supplier_cases(self):
        idx = self.eng.index
        toks = tokenize("subcontractor failing to deliver contracted components")
        for scores in (idx.bm25_scores(toks), idx.dense_scores(toks)):
            top = top_k_indices(scores, 10)
            labels = [idx.label_of("critical_issue", r) for r in top]
            self.assertGreaterEqual(labels.count("Supplier underperformance"), 6)

    def test_unknown_words_score_zero(self):
        scores = self.eng.index.bm25_scores(["zzzqqq"])
        self.assertEqual(float(scores.max()), 0.0)

    def test_retrieval_is_fast(self):
        start = time.perf_counter()
        for _ in range(50):
            self.eng.retriever.retrieve("milestones keep slipping on our rail upgrade")
        per_query = sub(time.perf_counter(), start) / 50 * 1000
        self.assertLess(per_query, 50.0)

    def test_filters_are_respected(self):
        res = self.eng.retriever.retrieve("costs are spiralling", filters={"industry": "Energy"})
        labels = {self.eng.index.label_of("industry", r) for r in res.candidates[:50]}
        self.assertEqual(labels, {"Energy"})


if __name__ == "__main__":
    unittest.main()
