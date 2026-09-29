"""Hybrid index construction and loading.

Two complementary representations of every project case are built:

1. A BM25 sparse matrix. Term weights are precomputed per document at build
   time, so scoring a query is a single sparse column slice and a matrix vector
   product, with no per query loops.
2. A dense latent semantic embedding (TF IDF with bigrams, reduced by truncated
   SVD). It captures synonymy such as "vendor" and "supplier" appearing in the
   same contexts, needs no GPU and no model download, and projects a query in
   microseconds.

Categorical and numeric metadata are stored as compact numpy arrays so filters
and evidence statistics are pure vector operations. Dense arrays are memory
mapped on load, which keeps cold start fast and lets several worker processes
share the same pages.
"""

import hashlib
import json
import time
from collections import Counter

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import svds

from .text import tokenize, with_bigrams
from .utils import sub
from .vocab import ISSUE_BY_NAME, ISSUES

CATEGORICAL = [
    "industry", "project_type", "region", "delivery_methodology", "contract_type",
    "organisation_size", "requirements_volatility", "regulatory_exposure", "critical_issue",
    "issue_severity", "detection_phase", "intervention_strategy", "recovery_success",
    "outcome_status", "primary_risk_category", "root_cause",
]
NUMERIC = [
    "schedule_ratio", "cost_ratio", "baseline_budget_gbp", "final_cost_gbp", "team_size",
    "technical_complexity", "sponsor_engagement", "pm_experience_years", "intervention_lead_time_days",
    "health_score", "customer_satisfaction", "benefits_realisation_pct", "quality_defect_rate",
    "vendor_count", "dependency_count", "stakeholder_count", "team_turnover_pct",
    "scope_change_requests", "planned_duration_days", "actual_duration_days",
    "cost_performance_index", "schedule_performance_index",
]


def document_text(row):
    """The searchable text of one case.

    Issue and root cause are repeated to weight them, and the document is expanded
    with the domain thesaurus for its issue (document expansion). This bridges the
    vocabulary gap between how cases are written up and how managers describe
    problems in their own words, for example "vendor" versus "supplier".
    """
    issue_key = ISSUE_BY_NAME.get(row["critical_issue"])
    thesaurus = " ".join(ISSUES[issue_key]["keywords"]) if issue_key else ""
    return " ".join([
        row["critical_issue"], row["critical_issue"], row["root_cause"], row["root_cause"],
        row["industry"], row["project_type"], row["delivery_methodology"], row["contract_type"],
        thesaurus, row["problem_statement"], row["resolution_narrative"], row["lessons_learned"],
    ])


def _count_matrix(token_lists, vocab=None, min_df=1):
    if vocab is None:
        df = Counter()
        for toks in token_lists:
            df.update(set(toks))
        terms = sorted(t for t, c in df.items() if c >= min_df)
        vocab = {t: i for i, t in enumerate(terms)}
    indptr, indices, data = [0], [], []
    for toks in token_lists:
        counts = Counter(t for t in toks if t in vocab)
        indices.extend(vocab[t] for t in counts)
        data.extend(counts.values())
        indptr.append(len(indices))
    matrix = sp.csr_matrix(
        (np.asarray(data, dtype=np.float32), np.asarray(indices, dtype=np.int32),
         np.asarray(indptr, dtype=np.int64)),
        shape=(len(token_lists), len(vocab)),
    )
    matrix.sum_duplicates()
    return matrix, vocab


def _row_ids(csr):
    return np.repeat(np.arange(csr.shape[0]), np.diff(csr.indptr))


def bm25_matrix(counts, k1, b):
    n_docs = counts.shape[0]
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
    idf = np.log(1.0 + (np.subtract(n_docs, df) + 0.5) / (df + 0.5))
    doc_len = np.asarray(counts.sum(axis=1)).ravel()
    avg_len = float(doc_len.mean())
    rows = _row_ids(counts)
    tf = counts.data.astype(np.float64)
    norm = sub(1.0, b) + b * doc_len[rows] / avg_len
    weights = tf * (k1 + 1.0) / (tf + k1 * norm) * idf[counts.indices]
    out = sp.csr_matrix((weights.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)
    return out.tocsc(), idf.astype(np.float32)


def tfidf_matrix(counts):
    n_docs = counts.shape[0]
    df = np.bincount(counts.indices, minlength=counts.shape[1]).astype(np.float64)
    idf = (np.log((1.0 + n_docs) / (1.0 + df)) + 1.0).astype(np.float32)
    data = (1.0 + np.log(counts.data)) * idf[counts.indices]
    mat = sp.csr_matrix((data.astype(np.float32), counts.indices, counts.indptr), shape=counts.shape)
    norms = np.sqrt(np.asarray(mat.multiply(mat).sum(axis=1)).ravel())
    norms[norms == 0] = 1.0
    mat = sp.diags(1.0 / norms) @ mat
    return mat.tocsr().astype(np.float32), idf


def _normalise_rows(x):
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return (x / norms).astype(np.float32)


def build(frame, out_dir, dims=160, k1=1.4, b=0.72, log=print):
    """Build every index artifact from the dataset frame and write it to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()

    texts = [document_text(r) for r in frame.to_dict("records")]
    tokens = [tokenize(t) for t in texts]
    log("  tokenised {} documents".format(len(tokens)))

    counts, vocab = _count_matrix(tokens, min_df=2)
    bm25, bm25_idf = bm25_matrix(counts, k1, b)
    sp.save_npz(out_dir / "bm25.npz", bm25)
    log("  BM25 matrix {} by {} with {} weights".format(bm25.shape[0], bm25.shape[1], bm25.nnz))

    grams = [with_bigrams(t) for t in tokens]
    dcounts, dvocab = _count_matrix(grams, min_df=3)
    tfidf, didf = tfidf_matrix(dcounts)
    k = int(min(dims, sub(min(tfidf.shape), 1)))
    u, s, vt = svds(tfidf, k=k, random_state=7)
    order = np.argsort(np.negative(s))
    u, s, vt = u[:, order], s[order], vt[order]
    embeddings = _normalise_rows(u * s)
    np.save(out_dir / "embeddings.npy", embeddings)
    np.save(out_dir / "components.npy", vt.astype(np.float32))
    np.save(out_dir / "dense_idf.npy", didf)
    explained = float((s ** 2).sum() / max(float(tfidf.multiply(tfidf).sum()), 1.0))
    log("  dense embeddings {} by {} ({:.1%} of TF IDF energy retained)".format(
        embeddings.shape[0], embeddings.shape[1], explained))

    labels, codes = {}, {}
    for col in CATEGORICAL:
        values = frame[col].astype(str)
        uniques = sorted(values.unique())
        lookup = {v: i for i, v in enumerate(uniques)}
        labels[col] = uniques
        codes[col] = values.map(lookup).to_numpy(dtype=np.int16)
    numeric = {col: frame[col].to_numpy(dtype=np.float32) for col in NUMERIC}
    np.savez(out_dir / "meta.npz", **{"c_" + k: v for k, v in codes.items()},
             **{"n_" + k: v for k, v in numeric.items()})

    with open(out_dir / "vocab.json", "w", encoding="utf8") as fh:
        json.dump({"bm25": vocab, "dense": dvocab}, fh)
    with open(out_dir / "labels.json", "w", encoding="utf8") as fh:
        json.dump(labels, fh)

    digest = hashlib.sha256(frame["project_id"].str.cat().encode()).hexdigest()[:16]
    manifest = {
        "documents": int(len(frame)), "bm25_terms": len(vocab), "dense_terms": len(dvocab),
        "embedding_dims": int(k), "energy_retained": round(explained, 4), "dataset_digest": digest,
        "build_seconds": round(sub(time.perf_counter(), started), 2),
        "project_ids": frame["project_id"].tolist(),
    }
    with open(out_dir / "manifest.json", "w", encoding="utf8") as fh:
        json.dump(manifest, fh)
    log("  index built in {:.2f}s".format(manifest["build_seconds"]))
    return manifest


class HybridIndex:
    """Read only view over the built artifacts."""

    def __init__(self, directory):
        with open(directory / "manifest.json", encoding="utf8") as fh:
            self.manifest = json.load(fh)
        with open(directory / "vocab.json", encoding="utf8") as fh:
            vocabs = json.load(fh)
        with open(directory / "labels.json", encoding="utf8") as fh:
            self.labels = json.load(fh)
        self.bm25_vocab = vocabs["bm25"]
        self.dense_vocab = vocabs["dense"]
        self.bm25 = sp.load_npz(directory / "bm25.npz").tocsc()
        self.embeddings = np.load(directory / "embeddings.npy", mmap_mode="r")
        self.components = np.load(directory / "components.npy")
        self.dense_idf = np.load(directory / "dense_idf.npy")
        meta = np.load(directory / "meta.npz")
        self.codes = {k[2:]: meta[k] for k in meta.files if k.startswith("c_")}
        self.numeric = {k[2:]: meta[k] for k in meta.files if k.startswith("n_")}
        self.project_ids = self.manifest["project_ids"]
        self.id_to_row = {pid: i for i, pid in enumerate(self.project_ids)}
        self.size = len(self.project_ids)
        self._components_t = np.ascontiguousarray(self.components.T)

    def label_of(self, column, row):
        return self.labels[column][int(self.codes[column][row])]

    def code_of(self, column, label):
        try:
            return self.labels[column].index(label)
        except ValueError:
            return None

    def bm25_scores(self, tokens):
        counts = Counter(t for t in tokens if t in self.bm25_vocab)
        if not counts:
            return np.zeros(self.size, dtype=np.float32)
        ids = np.fromiter((self.bm25_vocab[t] for t in counts), dtype=np.int64)
        weights = np.fromiter(counts.values(), dtype=np.float32)
        return np.asarray(self.bm25[:, ids] @ weights).ravel()

    def embed_query(self, tokens):
        counts = Counter(g for g in with_bigrams(tokens) if g in self.dense_vocab)
        if not counts:
            return None
        ids = np.fromiter((self.dense_vocab[g] for g in counts), dtype=np.int64)
        tf = 1.0 + np.log(np.fromiter(counts.values(), dtype=np.float32))
        weights = tf * self.dense_idf[ids]
        weights /= max(float(np.linalg.norm(weights)), 1.0 / 1e9)
        vec = weights @ self._components_t[ids]
        norm = float(np.linalg.norm(vec))
        return (vec / norm).astype(np.float32) if norm > 0 else None

    def dense_scores(self, tokens):
        vec = self.embed_query(tokens)
        if vec is None:
            return np.zeros(self.size, dtype=np.float32)
        return np.asarray(self.embeddings @ vec)
