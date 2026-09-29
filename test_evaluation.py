import unittest

from gantry_ai import evaluate
from tests.helpers import engine


class EvaluationSmokeTest(unittest.TestCase):
    def test_small_benchmark(self):
        report = evaluate.run(engine(), n_queries=60, log=lambda *a: None)
        self.assertGreater(report["retrieval"]["hybrid"]["ndcg_at_10"], report["retrieval"]["dense"]["ndcg_at_10"])
        rec = report["recommendation"]
        self.assertGreater(rec["gantry_top1_strong_rate"], rec["naive_majority_vote_strong_rate"])
        self.assertEqual(report["grounding"]["verified_answer_rate"], 1.0)
        self.assertIn("<table>", evaluate.to_markdown(report))

    def test_queries_never_copy_narratives(self):
        eng = engine()
        narratives = set(eng.frame["problem_statement"].head(2000))
        for q in evaluate.build_queries(50):
            self.assertNotIn(q["text"], narratives)


if __name__ == "__main__":
    unittest.main()
