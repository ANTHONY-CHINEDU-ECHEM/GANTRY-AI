import re
import unittest

import pandas as pd

from gantry_ai.config import settings
from gantry_ai.generator import COLUMNS, generate
from gantry_ai.utils import FORBIDDEN_CHARS
from gantry_ai.vocab import ISSUE_NAMES, INTERVENTION_NAMES
from tests.helpers import engine


class DatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        engine()
        cls.frame = pd.read_csv(settings.dataset_path)

    def test_minimum_shape(self):
        rows, cols = self.frame.shape
        self.assertGreaterEqual(rows, 16000)
        self.assertGreaterEqual(cols, 45)
        self.assertEqual(list(self.frame.columns), COLUMNS)

    def test_ids_unique_and_complete(self):
        self.assertTrue(self.frame["project_id"].is_unique)
        self.assertFalse(self.frame.isnull().any().any())

    def test_no_dash_characters_in_any_cell(self):
        blob = self.frame.to_csv(index=False)
        for ch in FORBIDDEN_CHARS:
            self.assertNotIn(ch, blob)

    def test_ratios_positive_and_consistent(self):
        f = self.frame
        self.assertTrue((f["schedule_ratio"] > 0).all())
        self.assertTrue((f["cost_ratio"] > 0).all())
        ratio = f["final_cost_gbp"] / f["baseline_budget_gbp"]
        self.assertLess((ratio.sub(f["cost_ratio"])).abs().max(), 0.002)

    def test_dates_are_ordered(self):
        pattern = re.compile(r"^\d{4}/\d{2}/\d{2}$")
        for col in ("start_date", "planned_end_date", "actual_end_date"):
            self.assertTrue(self.frame[col].str.match(pattern).all())
        self.assertTrue((self.frame["start_date"] < self.frame["actual_end_date"]).all())

    def test_vocabularies(self):
        self.assertTrue(set(self.frame["critical_issue"]) <= set(ISSUE_NAMES))
        self.assertTrue(set(self.frame["intervention_strategy"]) <= set(INTERVENTION_NAMES))
        self.assertEqual(self.frame["critical_issue"].nunique(), 15)

    def test_narratives_quote_their_own_numbers(self):
        sample = self.frame[self.frame["outcome_status"] != "Cancelled"].head(300)
        for _, row in sample.iterrows():
            self.assertIn("{:.0f}".format(100 * row["schedule_ratio"]), row["resolution_narrative"])

    def test_generator_is_deterministic(self):
        a = generate(300, seed=5)
        b = generate(300, seed=5)
        pd.testing.assert_frame_equal(a, b)

    def test_efficacy_signal_present(self):
        f = self.frame
        full = f["recovery_success"].eq("Full Recovery")
        harmful = f["intervention_strategy"].isin(["Sustained overtime push", "Adding general headcount to the team"])
        self.assertGreater(full[~harmful].mean(), 2 * full[harmful].mean())


if __name__ == "__main__":
    unittest.main()
