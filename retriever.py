"""Query understanding and hybrid retrieval.

Pipeline for one question:

1. Analyse the query: detect likely critical issues from the domain thesaurus
   and pick up industry, methodology and contract context.
2. Expand the query with the canonical names of the detected issues.
3. Score every case with BM25 and with dense semantic similarity, each in well
   under a millisecond.
4. Fuse the two rankings with Reciprocal Rank Fusion, which is robust to the
   very different score scales of the two retrievers.
5. Apply soft metadata boosts from the analysis and hard filters supplied by
   the caller.
6. Diagnose the problem with a similarity weighted vote over the top cases.
7. Select a diverse context set with Maximal Marginal Relevance so the answer
   is grounded in several distinct precedents rather than near duplicates.
"""

import re
from dataclasses import dataclass, field
from functools import lru_cache

import numpy as np

from .text import tokenize
from .utils import sub, top_k_indices
from .vocab import INTERVENTIONS, ISSUES, ISSUE_KEYS

INDUSTRY_SYNONYMS = {
    "Construction": ["construction", "building site", "contractor site", "civil engineering", "housing", "office build"],
    "Software": ["software", "saas", "mobile app", "codebase", "cloud migration", "erp", "web platform"],
    "Healthcare": ["healthcare", "hospital", "nhs", "clinical", "patient", "health"],
    "Financial Services": ["bank", "banking", "financial services", "payments", "insurance", "fintech", "trading"],
    "Energy": ["energy", "wind", "solar", "grid", "substation", "utility", "oil", "gas", "power"],
    "Manufacturing": ["manufacturing", "factory", "production line", "plant", "assembly", "robot"],
    "Public Sector": ["public sector", "government", "council", "ministry", "department", "citizen"],
    "Telecommunications": ["telecom", "telecoms", "telecommunications", "fibre", "fiber", "5g", "network rollout", "mobile network"],
    "Retail": ["retail", "store", "stores", "ecommerce", "shop", "omnichannel", "loyalty"],
    "Pharmaceuticals": ["pharma", "pharmaceutical", "clinical trial", "laboratory", "gmp", "drug"],
    "Aerospace and Defence": ["aerospace", "defence", "defense", "avionics", "aircraft", "military"],
    "Logistics": ["logistics", "warehouse", "fleet", "distribution centre", "transport", "shipping"],
}
METHOD_SYNONYMS = {
    "Agile Scrum": ["scrum", "agile", "sprint"], "Kanban": ["kanban"], "SAFe": ["safe framework", "scaled agile", "safe methodology", "agile release train"],
    "Waterfall": ["waterfall"], "PRINCE2": ["prince2", "prince 2"], "Hybrid": ["hybrid"],
    "Critical Chain": ["critical chain"], "Lean Construction": ["lean construction", "last planner"],
}
CONTRACT_SYNONYMS = {
    "Fixed Price": ["fixed price", "lump sum", "fixed fee"], "Time and Materials": ["time and materials"],
    "Cost Plus": ["cost plus", "cost reimbursable"], "Target Cost": ["target cost", "pain gain"],
    "Alliance": ["alliance", "alliancing"], "Internal Funding": ["internal project", "internally funded"],
}


@lru_cache(maxsize=4096)
def _pattern(phrase):
    return re.compile(r"(?<!\w)" + re.escape(phrase) + r"(?!\w)")


def _phrase_hits(text, phrases):
    hits = 0
    for phrase in phrases:
        p = phrase.strip()
        if p and p in text and _pattern(p).search(text):
            hits += 1 + p.count(" ")
    return hits


def _detect(text, table):
    scores = {k: _phrase_hits(text, v) for k, v in table.items()}
    best = max(scores, key=scores.get) if scores else None
    return best if best and scores[best] > 0 else None


@dataclass
class QueryAnalysis:
    text: str
    tokens: list
    issue_prior: np.ndarray
    detected_issues: list
    industry: str = None
    methodology: str = None
    contract: str = None
    mentioned_interventions: list = field(default_factory=list)

    def as_dict(self):
        return {
            "detected_issues": [ISSUES[k]["name"] for k in self.detected_issues],
            "industry": self.industry, "methodology": self.methodology, "contract": self.contract,
            "mentioned_interventions": [INTERVENTIONS[k]["name"] for k in self.mentioned_interventions],
        }


@lru_cache(maxsize=1)
def _stemmed_issue_keywords():
    """Issue thesaurus normalised with the same tokenizer as queries, so "resigning"
    matches "resigned" and "vendors" matches "vendor"."""
    table = {}
    for key in ISSUE_KEYS:
        phrases = ISSUES[key]["keywords"] + [ISSUES[key]["name"].lower()]
        table[key] = sorted({" ".join(tokenize(p)) for p in phrases if tokenize(p)})
    return table


def analyse(question):
    text = " " + question.lower() + " "
    stemmed = " " + " ".join(tokenize(question)) + " "
    table = _stemmed_issue_keywords()
    hits = np.array([_phrase_hits(stemmed, table[k]) for k in ISSUE_KEYS], dtype=float)
    prior = hits / hits.sum() if hits.sum() > 0 else np.zeros(len(ISSUE_KEYS))
    order = np.argsort(np.negative(hits), kind="stable")
    detected = [ISSUE_KEYS[i] for i in order[:3] if hits[i] > 0]
    mentioned = [k for k, v in INTERVENTIONS.items() if _phrase_hits(text, v["keywords"]) > 0]
    return QueryAnalysis(
        text=question, tokens=tokenize(question), issue_prior=prior, detected_issues=detected,
        industry=_detect(text, INDUSTRY_SYNONYMS), methodology=_detect(text, METHOD_SYNONYMS),
        contract=_detect(text, CONTRACT_SYNONYMS), mentioned_interventions=mentioned,
    )


@dataclass
class RetrievalResult:
    analysis: QueryAnalysis
    candidates: np.ndarray
    fused: np.ndarray
    bm25_top: np.ndarray
    dense_top: np.ndarray
    agreement: float
    diagnosis: list
    context: np.ndarray


class HybridRetriever:
    def __init__(self, index, settings):
        self.index = index
        self.settings = settings
        issue_labels = index.labels["critical_issue"]
        self._issue_code_to_key = np.array(
            [ISSUE_KEYS.index(next(k for k in ISSUE_KEYS if ISSUES[k]["name"] == lab)) for lab in issue_labels])

    def _mask(self, filters):
        mask = np.ones(self.index.size, dtype=bool)
        for column, wanted in (filters or {}).items():
            if column not in self.index.codes or wanted in (None, "", []):
                continue
            values = wanted if isinstance(wanted, (list, tuple, set)) else [wanted]
            codes = [c for c in (self.index.code_of(column, v) for v in values) if c is not None]
            mask &= np.isin(self.index.codes[column], codes)
        return mask

    def issue_keys_of(self, rows):
        return self._issue_code_to_key[self.index.codes["critical_issue"][rows]]

    def retrieve(self, question, filters=None, k=None, pool=None, analysis=None, mode="hybrid"):
        s = self.settings
        k = k or s.context_cases
        pool = pool or s.candidate_pool
        qa = analysis or analyse(question)
        expansion = []
        if mode == "hybrid":
            for key in qa.detected_issues[:2]:
                expansion.extend(tokenize(ISSUES[key]["name"]))
        tokens = qa.tokens + expansion

        mask = self._mask(filters)
        floor = np.float32(np.negative(1e9))
        rrf_k = float(s.rrf_k)
        ranks = np.arange(1, pool + 1, dtype=np.float32)
        fused = np.zeros(self.index.size, dtype=np.float32)
        bm_top = de_top = np.zeros(0, dtype=np.int64)

        if mode in ("hybrid", "fusion", "bm25"):
            bm = np.where(mask, self.index.bm25_scores(tokens), floor)
            bm_top = top_k_indices(bm, pool)
            bm_top = bm_top[bm[bm_top] > 0]
            fused[bm_top] += 1.0 / (rrf_k + ranks[:len(bm_top)])
        if mode in ("hybrid", "fusion", "dense"):
            de = np.where(mask, self.index.dense_scores(tokens), floor)
            de_top = top_k_indices(de, pool)
            de_top = de_top[de[de_top] > 0]
            fused[de_top] += 1.0 / (rrf_k + ranks[:len(de_top)])

        if mode == "hybrid":
            issue_code_prior = qa.issue_prior[self._issue_code_to_key]
            boost = 1.0 + 0.6 * issue_code_prior[self.index.codes["critical_issue"]]
            if qa.industry:
                code = self.index.code_of("industry", qa.industry)
                boost = boost + 0.15 * (self.index.codes["industry"] == code)
            if qa.methodology:
                code = self.index.code_of("delivery_methodology", qa.methodology)
                boost = boost + 0.06 * (self.index.codes["delivery_methodology"] == code)
            if qa.contract:
                code = self.index.code_of("contract_type", qa.contract)
                boost = boost + 0.05 * (self.index.codes["contract_type"] == code)
            fused = fused * boost.astype(np.float32)
        fused = np.where(mask, fused, 0.0).astype(np.float32)

        candidates = top_k_indices(fused, pool)
        candidates = candidates[fused[candidates] > 0]
        top_n = 50
        agreement = 0.0
        if len(bm_top) and len(de_top):
            agreement = len(np.intersect1d(bm_top[:top_n], de_top[:top_n])) / float(top_n)

        diagnosis = self._diagnose(candidates, fused, qa)
        context = self._mmr(candidates[:60], fused, k)
        return RetrievalResult(qa, candidates, fused, bm_top, de_top, agreement, diagnosis, context)

    def _diagnose(self, candidates, fused, qa, top=25):
        """Similarity weighted vote over the top cases, blended with the keyword prior."""
        head = candidates[:top]
        if len(head) == 0:
            order = np.argsort(np.negative(qa.issue_prior))
            return [(ISSUE_KEYS[i], float(qa.issue_prior[i])) for i in order[:2] if qa.issue_prior[i] > 0]
        weights = fused[head].astype(float)
        keys = self.issue_keys_of(head)
        vote = np.bincount(keys, weights=weights, minlength=len(ISSUE_KEYS))
        vote = vote / max(vote.sum(), 1.0 / 1e9)
        share = 0.7 * vote + 0.3 * qa.issue_prior if qa.issue_prior.sum() > 0 else vote
        order = np.argsort(np.negative(share), kind="stable")
        result = [(ISSUE_KEYS[order[0]], float(share[order[0]]))]
        for nxt in order[1:3]:
            if share[nxt] >= 0.18:
                result.append((ISSUE_KEYS[nxt], float(share[nxt])))
        return result

    def _mmr(self, candidates, fused, k):
        """Maximal Marginal Relevance over the dense embeddings."""
        if len(candidates) <= k:
            return candidates
        lam = self.settings.mmr_lambda
        emb = np.asarray(self.index.embeddings[candidates])
        rel = fused[candidates].astype(float)
        rel = rel / max(float(rel.max()), 1.0 / 1e9)
        sims = emb @ emb.T
        chosen = [0]
        max_sim = sims[0].copy()
        for _ in range(sub(k, 1)):
            score = np.subtract(lam * rel, sub(1.0, lam) * max_sim)
            score[chosen] = np.negative(np.inf)
            nxt = int(np.argmax(score))
            chosen.append(nxt)
            max_sim = np.maximum(max_sim, sims[nxt])
        return candidates[chosen]
