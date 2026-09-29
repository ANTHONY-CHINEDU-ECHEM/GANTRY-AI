import os
import unittest

from fastapi.testclient import TestClient

from gantry_ai.api import app
from gantry_ai.config import settings
from gantry_ai.llm import generate_answer, verify
from gantry_ai.risk_model import auc
from tests.helpers import engine


class FabricatingProvider:
    name = "fake"

    def generate(self, question, result, pack):
        return "Use overtime. It worked in 97% of cases [PRJ999999]."


class BrokenProvider:
    name = "broken"

    def generate(self, question, result, pack):
        raise RuntimeError("network down")


class VerifierTests(unittest.TestCase):
    def test_detects_invented_ids_and_numbers(self):
        check = verify("See [PRJ000001] and [PRJ123456]; success 88%.", "evidence PRJ000001 71%", ["PRJ000001"])
        self.assertEqual(check["invalid_ids"], ["PRJ123456"])
        self.assertEqual(check["unsupported_percentages"], [88.0])
        self.assertFalse(check["passed"])

    def test_strict_mode_replaces_fabricated_answer(self):
        result = engine().ask("Scope keeps growing on our software project", provider="extractive")
        out = generate_answer("q", result, settings, provider=FabricatingProvider(), strict=True)
        self.assertEqual(out["provider"], "extractive")
        self.assertTrue(out["verification"]["passed"])
        self.assertTrue(out["notes"])

    def test_provider_failure_falls_back(self):
        result = engine().ask("Scope keeps growing on our software project", provider="extractive")
        out = generate_answer("q", result, settings, provider=BrokenProvider())
        self.assertEqual(out["provider"], "extractive")

    def test_auc(self):
        self.assertAlmostEqual(auc(__import__("numpy").array([0, 0, 1, 1]), __import__("numpy").array([0.1, 0.2, 0.3, 0.4])), 1.0)


class ApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.pop("GANTRY_API_KEY", None)
        cls.client = TestClient(app)
        cls.client.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.client.__exit__(None, None, None)

    def test_health(self):
        body = self.client.get("/health").json()
        self.assertEqual(body["status"], "ok")
        self.assertGreaterEqual(body["cases"], 16000)

    def test_ask(self):
        r = self.client.post("/ask", json={"question": "Users refuse to adopt the new system", "k": 5})
        self.assertEqual(r.status_code, 200)
        self.assertEqual(r.json()["diagnosis"][0]["issue"], "Change resistance")

    def test_validation(self):
        self.assertEqual(self.client.post("/ask", json={"question": ""}).status_code, 422)
        self.assertEqual(self.client.post("/search", json={"query": "x y", "mode": "magic"}).status_code, 422)

    def test_search_recommend_assess_case(self):
        self.assertEqual(len(self.client.post("/search", json={"query": "regulator approval", "k": 4}).json()["results"]), 4)
        self.assertEqual(self.client.post("/recommend", json={"issue": "Budget overrun"}).status_code, 200)
        self.assertEqual(self.client.post("/recommend", json={"issue": "Unknown"}).status_code, 404)
        self.assertEqual(self.client.post("/assess", json={"technical_complexity": 9}).status_code, 200)
        self.assertEqual(self.client.get("/cases/PRJ000010").json()["project_id"], "PRJ000010")
        self.assertEqual(self.client.get("/cases/PRJ999999").status_code, 404)

    def test_bearer_auth(self):
        os.environ["GANTRY_API_KEY"] = "secret"
        try:
            self.assertEqual(self.client.get("/stats").status_code, 401)
            ok = self.client.get("/stats", headers={"Authorization": "Bearer secret"})
            self.assertEqual(ok.status_code, 200)
            self.assertEqual(self.client.get("/health").status_code, 200)
        finally:
            os.environ.pop("GANTRY_API_KEY", None)


if __name__ == "__main__":
    unittest.main()
