"""Instruction tuning export.

Turns the case base into supervised fine tuning data in the widely used chat
messages JSONL format. Two record types are produced:

grounded
    The user turn is a full evidence pack and the assistant turn is the
    verified extractive answer. Training on these teaches any open model to
    answer strictly from retrieved evidence with correct citations, which is
    the behaviour a RAG product needs.
case
    The user turn describes a single project problem and the assistant turn
    gives the documented response, result and lesson. This distils the case
    base itself into a model.
"""

import json
import random

from .answer import SYSTEM_PROMPT, evidence_pack
from .evaluate import build_queries
from .utils import clean_dashes

CASE_SYSTEM = ("You are Gantry AI, a senior project delivery advisor. Explain how a project problem was "
               "handled, what happened and what future teams should learn. Never use hyphens or dashes.")


def grounded_records(engine, limit=1000, seed=7):
    rng = random.Random(seed)
    questions = [q["text"] for q in build_queries(limit // 2, seed)]
    sample_rows = rng.sample(range(engine.index.size), sub_half(limit))
    for row in sample_rows:
        case = engine.case(row)
        questions.append("Here is my situation: " + case["problem_statement"] + " What should I do?")
    for q in questions:
        result = engine.ask(q, provider="extractive")
        if not result["verification"]["passed"]:
            continue
        yield {
            "type": "grounded",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": evidence_pack(q, result)},
                {"role": "assistant", "content": result["answer"]},
            ],
        }


def sub_half(limit):
    return int(limit) // 2 + int(limit) % 2


def case_records(engine, limit=None):
    rows = range(engine.index.size) if limit is None else range(min(int(limit), engine.index.size))
    for row in rows:
        c = engine.case(row)
        prompt = ("{name} is a {industry} project run with {method} under a {contract} contract. {problem} "
                  "How was this handled and what should we learn?").format(
            name=c["project_name"], industry=c["industry"], method=c["delivery_methodology"],
            contract=c["contract_type"].lower(), problem=c["problem_statement"])
        reply = "{} {} [{}]".format(c["resolution_narrative"], c["lessons_learned"], c["project_id"])
        yield {
            "type": "case",
            "messages": [
                {"role": "system", "content": CASE_SYSTEM},
                {"role": "user", "content": clean_dashes(prompt)},
                {"role": "assistant", "content": clean_dashes(reply)},
            ],
        }


def export(engine, path, grounded=1000, cases=None, log=print):
    path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"grounded": 0, "case": 0}
    with open(path, "w", encoding="utf8") as fh:
        for rec in grounded_records(engine, grounded):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["grounded"] += 1
        for rec in case_records(engine, cases):
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            counts["case"] += 1
    log("  wrote {grounded} grounded and {case} case records".format(**counts))
    return counts
