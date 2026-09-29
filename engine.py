"""The Gantry engine.

A single object that owns the loaded index, the retriever, the risk model and
the case store, and exposes the operations used by the command line, the API
and the web interface: ask, search, recommend, assess, case lookup and stats.
Loading takes a fraction of a second because dense arrays are memory mapped,
and answers to repeated questions are served from an in process LRU cache.
"""

import json
import threading
from collections import OrderedDict

import numpy as np
import pandas as pd

from . import evidence
from . import index as index_mod
from .config import settings as default_settings
from .generator import generate, write_ground_truth
from .llm import generate_answer
from .retriever import HybridRetriever, analyse
from .risk_model import RiskModel
from .schema import data_dictionary_markdown
from .utils import Stopwatch, safe_float
from .vocab import EFFICACY_MAP, INTERVENTIONS, ISSUES, ISSUE_BY_NAME

CASE_FIELDS = [
    "project_id", "project_name", "industry", "project_type", "region", "delivery_methodology",
    "contract_type", "organisation_size", "critical_issue", "issue_severity", "root_cause",
    "intervention_strategy", "intervention_lead_time_days", "recovery_success", "outcome_status",
    "schedule_ratio", "cost_ratio", "baseline_budget_gbp", "final_cost_gbp", "health_score",
    "problem_statement", "resolution_narrative", "lessons_learned",
]


def build_all(settings=default_settings, rows=None, seed=None, regenerate=False, log=print):
    """Generate the dataset if needed, then build the index and train the risk model."""
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.artifact_dir.mkdir(parents=True, exist_ok=True)
    path = settings.dataset_path
    if regenerate or not path.exists():
        log("Generating dataset")
        frame = generate(rows or settings.rows, seed if seed is not None else settings.seed)
        frame.to_csv(path, index=False)
        write_ground_truth(settings.ground_truth_path)
        settings.docs_dir.mkdir(parents=True, exist_ok=True)
        (settings.docs_dir / "DATA_DICTIONARY.md").write_text(data_dictionary_markdown(frame), encoding="utf8")
        log("  wrote {:,} rows and {} columns to {}".format(len(frame), len(frame.columns), path.name))
    else:
        frame = pd.read_csv(path)
    log("Building hybrid index")
    manifest = index_mod.build(frame, settings.artifact_dir, dims=settings.embedding_dims,
                               k1=settings.bm25_k1, b=settings.bm25_b, log=log)
    log("Training risk models")
    model = RiskModel.train(frame, log=log)
    model.save(settings.artifact_dir)
    return manifest


def artifacts_ready(settings=default_settings):
    needed = ["manifest.json", "bm25.npz", "embeddings.npy", "meta.npz", "risk_model.npz"]
    return settings.dataset_path.exists() and all((settings.artifact_dir / n).exists() for n in needed)


class GantryEngine:
    _instance = None
    _lock = threading.Lock()

    def __init__(self, settings=default_settings, auto_build=True, log=print):
        if not artifacts_ready(settings):
            if not auto_build:
                raise FileNotFoundError("Artifacts missing. Run: python gantry.py build")
            build_all(settings, log=log)
        self.settings = settings
        self.index = index_mod.HybridIndex(settings.artifact_dir)
        self.retriever = HybridRetriever(self.index, settings)
        self.risk = RiskModel.load(settings.artifact_dir)
        frame = pd.read_csv(settings.dataset_path)
        self.frame = frame
        self._records = frame[CASE_FIELDS].to_dict("records")
        self._cache = OrderedDict()
        self._cache_size = 256
        self._cache_lock = threading.Lock()

    @classmethod
    def shared(cls, **kwargs):
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls(**kwargs)
            return cls._instance

    def case(self, row_or_id):
        row = self.index.id_to_row.get(row_or_id) if isinstance(row_or_id, str) else int(row_or_id)
        if row is None:
            return None
        return dict(self._records[row])

    def _confidence(self, retrieval, blocks):
        reasons = []
        diag_share = retrieval.diagnosis[0][1] if retrieval.diagnosis else 0.0
        top_cases = blocks[0]["recommended"][0]["cases"] if blocks and blocks[0]["recommended"] else 0
        evidence_strength = min(1.0, top_cases / 150.0)
        agreement = retrieval.agreement
        score = 0.45 * evidence_strength + 0.35 * min(1.0, diag_share / 0.7) + 0.20 * agreement
        if retrieval.analysis.issue_prior.sum() == 0 and diag_share < 0.35:
            score *= 0.6
            reasons.append("The question does not name a clear delivery problem.")
        if top_cases:
            reasons.append("{} comparable cases support the top recommendation.".format(top_cases))
        reasons.append("Both retrieval methods agree on {:.0f}% of their top results.".format(100 * agreement))
        reasons.append("The diagnosis carries {:.0f}% of the weighted evidence.".format(100 * diag_share))
        label = "High" if score >= 0.7 else "Medium" if score >= 0.45 else "Low"
        return {"score": safe_float(score, 3), "label": label, "reasons": reasons}

    def _evidence_blocks(self, diagnosis, analysis, filters):
        blocks = []
        for issue_key, share in diagnosis[:2]:
            ref = evidence.reference_class(self.index, issue_key, analysis, filters,
                                           self.settings.evidence_min_cases)
            stats, baseline = evidence.intervention_stats(self.index, ref.rows, issue_key)
            recommended, avoid = evidence.split_recommendations(stats, baseline)
            strong_names = [s["intervention"] for s in recommended[:2]]
            blocks.append({
                "issue": ISSUES[issue_key]["name"], "issue_key": issue_key, "share": safe_float(share, 3),
                "reference_class": ref.description, "narrowed_by": ref.narrowed_by,
                "baseline_full_recovery": safe_float(baseline, 4),
                "recommended": recommended, "avoid": avoid, "all_interventions": stats,
                "timing": evidence.timing_effect(self.index, ref.rows, strong_names),
                "outlook": evidence.outlook(self.index, ref.rows),
                "_rows": ref.rows,
            })
        return blocks

    def _precedents(self, retrieval, blocks, context_rows):
        rows = list(context_rows)
        seen = set(rows)
        full = self.index.code_of("recovery_success", "Full Recovery")
        for block in blocks:
            issue_code = self.index.code_of("critical_issue", block["issue"])
            for rec in block["recommended"]:
                code = self.index.code_of("intervention_strategy", rec["intervention"])
                pool = retrieval.candidates
                match = pool[(self.index.codes["intervention_strategy"][pool] == code)
                             & (self.index.codes["recovery_success"][pool] == full)
                             & (self.index.codes["critical_issue"][pool] == issue_code)]
                if len(match) == 0:
                    ref = block["_rows"]
                    match = ref[(self.index.codes["intervention_strategy"][ref] == code)
                                & (self.index.codes["recovery_success"][ref] == full)]
                if len(match) and int(match[0]) not in seen:
                    rows.append(int(match[0]))
                    seen.add(int(match[0]))
        return rows

    def ask(self, question, filters=None, k=None, provider=None, strict=True):
        key = json.dumps([question.strip().lower(), filters or {}, k, provider, strict], sort_keys=True)
        with self._cache_lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                cached = dict(self._cache[key])
                cached["cached"] = True
                return cached
        watch = Stopwatch()
        with watch.lap("analyse"):
            analysis = analyse(question)
        with watch.lap("retrieve"):
            retrieval = self.retriever.retrieve(question, filters=filters, k=k, analysis=analysis)
        with watch.lap("evidence"):
            blocks = self._evidence_blocks(retrieval.diagnosis, analysis, filters)
            rows = self._precedents(retrieval, blocks, retrieval.context)
            relevance = retrieval.fused
            top_score = float(relevance[retrieval.candidates[0]]) if len(retrieval.candidates) else 1.0
            cases = []
            for r in rows:
                c = self.case(int(r))
                c["relevance"] = safe_float(float(relevance[r]) / top_score if top_score else 0.0, 3)
                cases.append(c)
        result = {
            "question": question,
            "analysis": analysis.as_dict(),
            "diagnosis": [{"issue": ISSUES[k_]["name"], "share": safe_float(s, 3)} for k_, s in retrieval.diagnosis],
            "recommendations": [{k_: v for k_, v in b.items() if k_ != "_rows"} for b in blocks],
            "cases": cases,
            "retrieval": {"candidates": int(len(retrieval.candidates)),
                          "bm25_dense_agreement": safe_float(retrieval.agreement, 3)},
        }
        result["confidence"] = self._confidence(retrieval, result["recommendations"])
        with watch.lap("generate"):
            gen = generate_answer(question, result, self.settings, provider=provider, strict=strict)
        result.update({"answer": gen["answer"], "provider": gen["provider"],
                       "verification": gen["verification"], "notes": gen["notes"]})
        result["timings_ms"] = dict(watch.timings, total=watch.total())
        result["cached"] = False
        with self._cache_lock:
            self._cache[key] = result
            if len(self._cache) > self._cache_size:
                self._cache.popitem(last=False)
        return result

    def clear_cache(self):
        with self._cache_lock:
            self._cache.clear()

    def evidence_pack(self, question, filters=None):
        from .answer import evidence_pack
        result = self.ask(question, filters=filters, provider="extractive")
        return evidence_pack(question, result)

    def search(self, query, filters=None, k=10, mode="hybrid"):
        watch = Stopwatch()
        with watch.lap("retrieve"):
            retrieval = self.retriever.retrieve(query, filters=filters, k=k, mode=mode)
        top = retrieval.candidates[:k]
        best = float(retrieval.fused[top[0]]) if len(top) else 1.0
        hits = []
        for r in top:
            c = self.case(int(r))
            c["score"] = safe_float(float(retrieval.fused[r]) / best if best else 0.0, 4)
            hits.append(c)
        return {"query": query, "mode": mode, "results": hits, "timings_ms": watch.timings}

    def recommend(self, issue, industry=None, methodology=None, contract=None, filters=None):
        issue_key = ISSUE_BY_NAME.get(issue) or (issue if issue in ISSUES else None)
        if issue_key is None:
            raise KeyError("Unknown issue: {}".format(issue))
        analysis = analyse("")
        analysis.industry, analysis.methodology, analysis.contract = industry, methodology, contract
        block = self._evidence_blocks([(issue_key, 1.0)], analysis, filters)[0]
        block.pop("_rows", None)
        return block

    def assess(self, profile):
        prediction = self.risk.predict(profile)
        playbook = []
        for item in prediction["likely_issues"][:3]:
            key = ISSUE_BY_NAME[item["issue"]]
            block = self.recommend(item["issue"], industry=profile.get("industry"))
            best = block["recommended"][0]["intervention"] if block["recommended"] else None
            playbook.append({"issue": item["issue"], "probability": item["probability"],
                             "prepare": best, "reference_class": block["reference_class"],
                             "early_warning_signs": ISSUES[key]["keywords"][:5]})
        prediction["playbook"] = playbook
        prediction["model_metrics"] = self.risk.metrics
        return prediction

    def stats(self):
        f = self.frame
        return {
            "cases": int(len(f)), "columns": int(len(f.columns)),
            "index": {k: v for k, v in self.index.manifest.items() if k != "project_ids"},
            "industries": f["industry"].value_counts().to_dict(),
            "issues": f["critical_issue"].value_counts().to_dict(),
            "outcomes": f["outcome_status"].value_counts().to_dict(),
            "recovery": f["recovery_success"].value_counts().to_dict(),
            "median_cost_ratio": safe_float(f["cost_ratio"].median(), 3),
            "median_schedule_ratio": safe_float(f["schedule_ratio"].median(), 3),
            "total_portfolio_value_gbp": int(f["baseline_budget_gbp"].sum()),
        }

    def options(self):
        return {
            "industries": self.index.labels["industry"],
            "methodologies": self.index.labels["delivery_methodology"],
            "contracts": self.index.labels["contract_type"],
            "organisation_sizes": self.index.labels["organisation_size"],
            "regions": self.index.labels["region"],
            "issues": [ISSUES[k]["name"] for k in ISSUES],
            "interventions": [INTERVENTIONS[k]["name"] for k in INTERVENTIONS],
            "volatility": ["Low", "Moderate", "High", "Severe"],
            "regulatory": ["Low", "Medium", "High"],
        }


__all__ = ["GantryEngine", "build_all", "artifacts_ready", "EFFICACY_MAP", "np"]
