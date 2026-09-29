import unittest

from gantry_ai.vocab import EFFICACY_MAP, INTERVENTIONS, ISSUES
from tests.helpers import engine


class EvidenceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()

    def test_top_recommendation_is_strong_for_every_issue(self):
        for key, issue in ISSUES.items():
            block = self.eng.recommend(issue["name"])
            top = block["recommended"][0]["intervention"]
            strong = [INTERVENTIONS[k]["name"] for k in EFFICACY_MAP[key]["strong"]]
            self.assertIn(top, strong, "{} got {}".format(issue["name"], top))

    def test_harmful_interventions_never_recommended(self):
        for key, issue in ISSUES.items():
            block = self.eng.recommend(issue["name"])
            harmful = {INTERVENTIONS[k]["name"] for k in EFFICACY_MAP[key]["harmful"]}
            recommended = {s["intervention"] for s in block["recommended"]}
            self.assertFalse(recommended & harmful)

    def test_reference_class_narrows_only_when_large_enough(self):
        block = self.eng.recommend("Technical debt accumulation", industry="Software")
        self.assertIn("Software", block["narrowed_by"])
        self.assertGreaterEqual(int(block["reference_class"].split()[0]), 200)


class AskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.eng = engine()
        cls.result = cls.eng.ask("Our vendor keeps missing delivery dates on a fixed price construction job")

    def test_diagnosis(self):
        self.assertEqual(self.result["diagnosis"][0]["issue"], "Supplier underperformance")

    def test_answer_is_grounded(self):
        v = self.result["verification"]
        self.assertTrue(v["passed"])
        self.assertEqual(v["invalid_ids"], [])
        self.assertEqual(v["citation_precision"], 1.0)

    def test_answer_structure(self):
        for heading in ("### Diagnosis", "### Recommended actions", "### Confidence"):
            self.assertIn(heading, self.result["answer"])

    def test_context_detection(self):
        self.assertEqual(self.result["analysis"]["industry"], "Construction")
        self.assertEqual(self.result["analysis"]["contract"], "Fixed Price")

    def test_cache(self):
        again = self.eng.ask("Our vendor keeps missing delivery dates on a fixed price construction job")
        self.assertTrue(again["cached"])

    def test_multi_issue_question(self):
        r = self.eng.ask("We are weeks behind schedule and our lead engineer resigned last week")
        names = {d["issue"] for d in r["diagnosis"]}
        self.assertTrue({"Critical path slippage", "Key person dependency"} & names)

    def test_off_topic_question_is_handled(self):
        r = self.eng.ask("What is the capital of France?")
        self.assertEqual(r["confidence"]["label"], "Low")

    def test_assess(self):
        out = self.eng.assess({"industry": "Software", "technical_complexity": 9, "sponsor_engagement": 1})
        for risk in out["risks"].values():
            self.assertTrue(0.0 < risk["probability"] < 1.0)
        self.assertEqual(len(out["playbook"]), 3)


if __name__ == "__main__":
    unittest.main()
