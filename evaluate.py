"""Evaluation harness.

Measures Gantry AI against the ground truth the generator knows:

Retrieval quality
    Precision at 10, MRR and nDCG at 10 for BM25 alone, dense alone, plain
    rank fusion, and the full hybrid pipeline with query understanding.
Diagnosis
    How often the engine names the true critical issue.
Recommendation quality
    How often the top recommendation is a genuinely strong intervention for the
    true issue, compared with two naive RAG baselines that copy what similar
    projects did most often, or what most often succeeded.
Grounding
    Share of answers whose citations and percentages all verify.
Latency
    End to end answer time percentiles with the cache disabled.

Queries are written in phrasing that never appears in the case narratives, so
the benchmark tests understanding rather than string matching.
"""

import json
import random
import time

import numpy as np

from .utils import safe_float, sub
from .vocab import EFFICACY_MAP, INDUSTRIES, INTERVENTIONS, ISSUES, ISSUE_KEYS

OPENERS = [
    "We are delivering a {ptype} in {industry} and {q}.",
    "On our {industry} {ptype}, {q}. What should we do?",
    "{Q} on a {ptype} for a {industry} client. How do we recover?",
    "Advice needed: {q}.",
]


def build_queries(n=400, seed=2026, multi_share=0.2):
    rng = random.Random(seed)
    industries = list(INDUSTRIES)
    queries = []
    for i in range(n):
        industry = rng.choice(industries)
        ptype = rng.choice(INDUSTRIES[industry]["types"]).lower()
        if rng.random() < multi_share:
            a, b = rng.sample(ISSUE_KEYS, 2)
            q = rng.choice(ISSUES[a]["queries"]) + " and " + rng.choice(ISSUES[b]["queries"])
            issues = [a, b]
        else:
            a = rng.choice(ISSUE_KEYS)
            q = rng.choice(ISSUES[a]["queries"])
            issues = [a]
        opener = rng.choice(OPENERS)
        text = opener.format(ptype=ptype, industry=industry, q=q, Q=q[:1].upper() + q[1:])
        uses_industry = "{industry}" in opener
        queries.append({"id": i, "text": text, "issues": issues,
                        "industry": industry if uses_industry else None})
    return queries


def _grades(engine, rows, query):
    issue_codes = {engine.index.code_of("critical_issue", ISSUES[k]["name"]) for k in query["issues"]}
    ind_code = engine.index.code_of("industry", query["industry"]) if query["industry"] else None
    ic = engine.index.codes["critical_issue"][rows]
    rel = np.isin(ic, list(issue_codes)).astype(float)
    if ind_code is not None:
        rel = rel + rel * (engine.index.codes["industry"][rows] == ind_code)
    return rel


def _ndcg(grades, k=10):
    g = grades[:k]
    discounts = 1.0 / np.log2(np.arange(2, len(g) + 2))
    dcg = float(((2 ** g) + np.negative(1.0)) @ discounts) if len(g) else 0.0
    ideal = np.flip(np.sort(g))
    idcg = float(((2 ** ideal) + np.negative(1.0)) @ discounts) if len(ideal) else 0.0
    return dcg / idcg if idcg > 0 else 0.0


def retrieval_metrics(engine, queries, modes=("bm25", "dense", "fusion", "hybrid"), k=10):
    out = {}
    for mode in modes:
        p_at, rr, nd, lat = [], [], [], []
        for q in queries:
            t = time.perf_counter()
            res = engine.retriever.retrieve(q["text"], k=k, mode=mode)
            lat.append(sub(time.perf_counter(), t) * 1000.0)
            rows = res.candidates[:k]
            grades = _grades(engine, rows, q)
            hits = grades > 0
            p_at.append(float(hits.mean()) if len(rows) else 0.0)
            first = np.flatnonzero(hits)
            rr.append(1.0 / (first[0] + 1) if len(first) else 0.0)
            nd.append(_ndcg(grades, k))
        out[mode] = {"precision_at_10": safe_float(np.mean(p_at)), "mrr": safe_float(np.mean(rr)),
                     "ndcg_at_10": safe_float(np.mean(nd)), "p50_ms": safe_float(np.median(lat), 3)}
    return out


def _tier(issue_key, intervention_name):
    key = next(k for k, v in INTERVENTIONS.items() if v["name"] == intervention_name)
    tiers = EFFICACY_MAP[issue_key]
    for t in ("strong", "moderate", "harmful"):
        if key in tiers[t]:
            return t
    return "neutral"


def _naive_baselines(engine, retrieval, top=25):
    head = retrieval.candidates[:top]
    if len(head) == 0:
        return None, None
    ints = engine.index.codes["intervention_strategy"][head]
    labels = engine.index.labels["intervention_strategy"]
    majority = labels[int(np.bincount(ints).argmax())]
    full = engine.index.code_of("recovery_success", "Full Recovery")
    wins = ints[engine.index.codes["recovery_success"][head] == full]
    success = labels[int(np.bincount(wins).argmax())] if len(wins) else majority
    return majority, success


def answer_metrics(engine, queries):
    diag_top1, diag_recall = [], []
    gantry_strong, gantry_good, gantry_harm = [], [], []
    maj_strong, succ_strong = [], []
    avoid_correct, avoid_total = 0, 0
    grounded, citation, latency = [], [], []
    for q in queries:
        engine.clear_cache()
        t = time.perf_counter()
        result = engine.ask(q["text"])
        latency.append(sub(time.perf_counter(), t) * 1000.0)
        truth = q["issues"]
        names = [d["issue"] for d in result["diagnosis"]]
        truth_names = [ISSUES[k]["name"] for k in truth]
        diag_top1.append(1.0 if names and names[0] in truth_names else 0.0)
        diag_recall.append(len(set(names) & set(truth_names)) / len(truth_names))
        grounded.append(1.0 if result["verification"]["passed"] else 0.0)
        citation.append(result["verification"]["citation_precision"])
        if len(truth) != 1 or not result["recommendations"]:
            continue
        block = result["recommendations"][0]
        if block["issue"] != truth_names[0] or not block["recommended"]:
            gantry_strong.append(0.0)
            gantry_good.append(0.0)
        else:
            tier = _tier(truth[0], block["recommended"][0]["intervention"])
            gantry_strong.append(1.0 if tier == "strong" else 0.0)
            gantry_good.append(1.0 if tier in ("strong", "moderate") else 0.0)
            gantry_harm.append(1.0 if any(_tier(truth[0], s["intervention"]) == "harmful"
                                          for s in block["recommended"]) else 0.0)
            for s in block["avoid"]:
                avoid_total += 1
                avoid_correct += 1 if _tier(truth[0], s["intervention"]) in ("harmful", "neutral") else 0
        retrieval = engine.retriever.retrieve(q["text"])
        majority, success = _naive_baselines(engine, retrieval)
        if majority:
            maj_strong.append(1.0 if _tier(truth[0], majority) == "strong" else 0.0)
            succ_strong.append(1.0 if _tier(truth[0], success) == "strong" else 0.0)
    lat = np.asarray(latency)
    return {
        "diagnosis": {"top1_accuracy": safe_float(np.mean(diag_top1)), "issue_recall": safe_float(np.mean(diag_recall))},
        "recommendation": {
            "gantry_top1_strong_rate": safe_float(np.mean(gantry_strong)),
            "gantry_top1_strong_or_moderate_rate": safe_float(np.mean(gantry_good)),
            "gantry_harmful_in_recommendations_rate": safe_float(np.mean(gantry_harm) if gantry_harm else 0.0),
            "avoid_list_precision": safe_float(avoid_correct / avoid_total if avoid_total else 1.0),
            "naive_majority_vote_strong_rate": safe_float(np.mean(maj_strong)),
            "naive_success_vote_strong_rate": safe_float(np.mean(succ_strong)),
        },
        "grounding": {"verified_answer_rate": safe_float(np.mean(grounded)),
                      "mean_citation_precision": safe_float(np.mean(citation))},
        "latency_ms": {"p50": safe_float(np.percentile(lat, 50), 2), "p95": safe_float(np.percentile(lat, 95), 2),
                       "p99": safe_float(np.percentile(lat, 99), 2), "mean": safe_float(lat.mean(), 2)},
    }


def run(engine, n_queries=400, seed=2026, log=print):
    queries = build_queries(n_queries, seed)
    log("Evaluating retrieval on {} queries".format(len(queries)))
    retrieval = retrieval_metrics(engine, queries)
    for mode, m in retrieval.items():
        log("  {:<7} P@10 {:.3f}  MRR {:.3f}  nDCG@10 {:.3f}  {:.2f} ms".format(
            mode, m["precision_at_10"], m["mrr"], m["ndcg_at_10"], m["p50_ms"]))
    log("Evaluating diagnosis, recommendations, grounding and latency")
    answers = answer_metrics(engine, queries)
    for section, values in answers.items():
        log("  {}: {}".format(section, values))
    report = {
        "queries": len(queries), "seed": seed, "dataset_cases": engine.index.size,
        "retrieval": retrieval, **answers, "risk_models": engine.risk.metrics,
        "examples": [q["text"] for q in queries[:5]],
    }
    return report


def _row(cells, tag="td"):
    return "<tr>" + "".join("<{0}>{1}</{0}>".format(tag, c) for c in cells) + "</tr>"


def to_markdown(report):
    r = report["retrieval"]
    names = {"bm25": "BM25 only", "dense": "Dense only", "fusion": "Rank fusion", "hybrid": "Full hybrid with query understanding"}
    ret_rows = [_row(["Method", "Precision at 10", "MRR", "nDCG at 10", "Median latency"], "th")]
    for mode in ("bm25", "dense", "fusion", "hybrid"):
        m = r[mode]
        ret_rows.append(_row([names[mode], "{:.3f}".format(m["precision_at_10"]), "{:.3f}".format(m["mrr"]),
                              "{:.3f}".format(m["ndcg_at_10"]), "{:.2f} ms".format(m["p50_ms"])]))
    rec = report["recommendation"]
    rec_rows = [_row(["Approach", "Top recommendation is a strong intervention"], "th"),
                _row(["Naive RAG: copy the most common response in similar cases",
                      "{:.1%}".format(rec["naive_majority_vote_strong_rate"])]),
                _row(["Naive RAG: copy the most frequently successful response",
                      "{:.1%}".format(rec["naive_success_vote_strong_rate"])]),
                _row(["Gantry evidence engine (Wilson ranked reference class)",
                      "{:.1%}".format(rec["gantry_top1_strong_rate"])])]
    lat = report["latency_ms"]
    risk = report["risk_models"]
    risk_rows = [_row(["Model", "Test metric"], "th")]
    for name, m in risk.items():
        if "test_auc" in m:
            risk_rows.append(_row([name.replace("_", " "), "AUC {:.3f} (base rate {:.1%})".format(m["test_auc"], m["base_rate"])]))
    ic = risk["issue_classifier"]
    risk_rows.append(_row(["likely issue classifier", "top 3 accuracy {:.1%} (chance top 1 {:.1%})".format(
        ic["top3_accuracy"], ic["chance_top1"])]))
    parts = [
        "# Evaluation report", "",
        "Generated by `python gantry.py evaluate` on {} benchmark queries against {:,} cases. "
        "Queries use phrasing that never appears in the case narratives, and {:.0%} of them describe two "
        "problems at once.".format(report["queries"], report["dataset_cases"], 0.2), "",
        "## Retrieval", "", "<table>", *ret_rows, "</table>", "",
        "## Diagnosis", "",
        "The primary diagnosis matched the true issue in {:.1%} of queries, and {:.1%} of all true issues "
        "were named somewhere in the diagnosis.".format(report["diagnosis"]["top1_accuracy"],
                                                        report["diagnosis"]["issue_recall"]), "",
        "## Recommendation quality", "", "<table>", *rec_rows, "</table>", "",
        "Top recommendation was strong or moderately effective in {:.1%} of cases. A harmful intervention "
        "appeared among recommendations in {:.1%} of answers, and {:.1%} of interventions on the avoid list "
        "were genuinely ineffective or harmful.".format(rec["gantry_top1_strong_or_moderate_rate"],
                                                     rec["gantry_harmful_in_recommendations_rate"],
                                                     rec["avoid_list_precision"]), "",
        "## Grounding", "",
        "{:.1%} of answers passed citation and numeric verification, with mean citation precision of "
        "{:.3f}.".format(report["grounding"]["verified_answer_rate"], report["grounding"]["mean_citation_precision"]), "",
        "## Latency", "",
        "End to end answer latency with the cache disabled, extractive provider: median {:.2f} ms, "
        "p95 {:.2f} ms, p99 {:.2f} ms.".format(lat["p50"], lat["p95"], lat["p99"]), "",
        "## Predictive risk models", "", "<table>", *risk_rows, "</table>", "",
    ]
    return "\n".join(parts)


def write(report, report_dir, docs_dir):
    report_dir.mkdir(parents=True, exist_ok=True)
    with open(report_dir / "evaluation.json", "w", encoding="utf8") as fh:
        json.dump(report, fh, indent=2)
    (docs_dir / "EVALUATION.md").write_text(to_markdown(report), encoding="utf8")
