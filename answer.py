"""Answer composition.

Two outputs are produced from the same structured evidence:

* an evidence pack, a compact plain text brief that a language model receives
  as its only source of truth, and
* an extractive answer, written deterministically from the evidence with no
  language model at all. It is the default provider, the fallback whenever a
  model call fails verification, and the reference the verifier compares
  against.
"""

from .utils import clean_dashes, fmt_money, pct
from .vocab import INTERVENTIONS, ISSUES

SYSTEM_PROMPT = """You are Gantry AI, a senior project delivery advisor.
You answer questions about troubled or complex projects using ONLY the evidence pack you are given.

Rules:
1. Base every recommendation on the evidence pack. Do not use outside knowledge to invent statistics.
2. Cite precedent cases by their project ID in square brackets, for example [PRJ000123]. Only cite IDs that appear in the evidence pack.
3. Quote numbers exactly as they appear in the evidence pack. Never invent or round differently.
4. If the evidence is thin or the question is outside project delivery, say so plainly.
5. Structure the answer with these headings: Diagnosis, Recommended actions, Avoid, Act quickly, What to expect, Confidence.
6. Write in British English, be concise and practical, and address the reader as the project lead.
7. Never use hyphens or dashes of any kind. Write compound words as separate words."""


def _case_line(case):
    return ("[{id}] {name} | {industry} | {method} | issue: {issue} | response: {intv} | "
            "result: {rec}, {out}, duration {s} of plan, cost {c} of budget").format(
        id=case["project_id"], name=case["project_name"], industry=case["industry"],
        method=case["delivery_methodology"], issue=case["critical_issue"],
        intv=case["intervention_strategy"], rec=case["recovery_success"], out=case["outcome_status"],
        s=pct(case["schedule_ratio"]), c=pct(case["cost_ratio"]))


def evidence_pack(question, result):
    lines = ["QUESTION", question, "", "DIAGNOSIS"]
    for d in result["diagnosis"]:
        lines.append("{} (share of evidence {})".format(d["issue"], pct(d["share"])))
    for block in result["recommendations"]:
        lines.append("")
        lines.append("REFERENCE CLASS FOR {}: {}".format(block["issue"].upper(), block["reference_class"]))
        lines.append("Baseline full recovery rate: {}".format(pct(block["baseline_full_recovery"])))
        for s in block["recommended"]:
            lines.append("RECOMMENDED {}: full recovery {} of {} cases, lower bound {}, median cost {} of budget, "
                         "median duration {} of plan, median lead time {} days".format(
                             s["intervention"], pct(s["full_recovery_rate"]), s["cases"],
                             pct(s["confidence_lower_bound"]), pct(s["median_cost_ratio"]),
                             pct(s["median_schedule_ratio"]), int(s["median_lead_time_days"])))
        for s in block["avoid"]:
            lines.append("AVOID {}: full recovery only {} of {} cases".format(
                s["intervention"], pct(s["full_recovery_rate"]), s["cases"]))
        t = block.get("timing")
        if t:
            lines.append("TIMING: starting within {} days gave full recovery {} versus {} when later".format(
                int(t["threshold_days"]), pct(t["early_full_recovery_rate"]), pct(t["late_full_recovery_rate"])))
        o = block.get("outlook") or {}
        if o:
            lines.append("OUTLOOK: median duration {} of plan, median cost {} of budget, P80 cost {}, "
                         "P80 duration {}, suggested contingency {:.0f}% on cost and {:.0f}% on schedule".format(
                             pct(o["median_schedule_ratio"]), pct(o["median_cost_ratio"]),
                             pct(o["p80_cost_ratio"]), pct(o["p80_schedule_ratio"]),
                             o["suggested_cost_contingency_pct"], o["suggested_schedule_contingency_pct"]))
    lines.append("")
    lines.append("PRECEDENT CASES")
    for case in result["cases"]:
        lines.append(_case_line(case))
        lines.append("  Problem: " + case["problem_statement"])
        lines.append("  Resolution: " + case["resolution_narrative"])
        lines.append("  Lesson: " + case["lessons_learned"])
    lines.append("")
    conf = result["confidence"]
    lines.append("CONFIDENCE: {} ({}). {}".format(conf["label"], conf["score"], " ".join(conf["reasons"])))
    return "\n".join(lines)


def _precedent_for(result, intervention_name):
    for case in result["cases"]:
        if case["intervention_strategy"] == intervention_name and case["recovery_success"] == "Full Recovery":
            return case["project_id"]
    return None


def extractive_answer(result):
    """Write a complete, cited answer directly from the evidence."""
    diag = result["diagnosis"]
    if not diag:
        return ("### Diagnosis\nThe question does not describe a recognisable project delivery problem, "
                "so there is no reliable evidence to draw on. Describe the symptoms you are seeing, for "
                "example missed milestones, supplier issues or rising costs, and ask again.")
    out = []
    primary = diag[0]
    cases = result["cases"]
    closest = ", ".join("[{}]".format(c["project_id"]) for c in cases[:2])
    text = "Your situation most closely matches **{}** ({} of the weighted evidence)".format(
        primary["issue"].lower(), pct(primary["share"]))
    if len(diag) > 1:
        text += ", with **{}** as a secondary pattern ({})".format(diag[1]["issue"].lower(), pct(diag[1]["share"]))
    text += ". The closest precedents are {}.".format(closest) if closest else "."
    out.append("### Diagnosis\n" + text)

    for block_no, block in enumerate(result["recommendations"]):
        heading = "### Recommended actions" if block_no == 0 else "### Also address {}".format(block["issue"].lower())
        items = []
        for i, s in enumerate(block["recommended"], 1):
            key = s["key"]
            precedent = _precedent_for(result, s["intervention"])
            line = ("{}. **{}.** Full recovery in {} of {} comparable cases (statistical lower bound {}), "
                    "against a {} baseline for this problem. Median final cost {} of budget and duration {} of plan. "
                    "How it works: {}.").format(
                i, s["intervention"], pct(s["full_recovery_rate"]), s["cases"], pct(s["confidence_lower_bound"]),
                pct(block["baseline_full_recovery"]), pct(s["median_cost_ratio"]), pct(s["median_schedule_ratio"]),
                INTERVENTIONS[key]["mechanism"])
            if precedent:
                line += " Precedent: [{}].".format(precedent)
            items.append(line)
        if not items:
            items.append("No single response stands out in this reference class; the evidence is too even to rank.")
        out.append(heading + "\n" + "\n".join(items))

        if block["avoid"]:
            avoid = ["**{}** recovered fully in only {} of {} comparable cases.".format(
                s["intervention"], pct(s["full_recovery_rate"]), s["cases"]) for s in block["avoid"]]
            out.append("### Avoid\n" + " ".join(avoid))

        t = block.get("timing")
        if t and block_no == 0 and t["early_full_recovery_rate"] > t["late_full_recovery_rate"]:
            out.append("### Act quickly\nAmong the strongest responses, starting within {} days of detection gave "
                       "full recovery in {} of cases, compared with {} when the response started later.".format(
                           int(t["threshold_days"]), pct(t["early_full_recovery_rate"]),
                           pct(t["late_full_recovery_rate"])))

        o = block.get("outlook") or {}
        if o and block_no == 0:
            out.append("### What to expect\nIn this reference class ({}), the median project finished at {} of "
                       "planned duration and {} of budget. Holding P80 contingency means about {} on cost and "
                       "{} on schedule from the point of detection.".format(
                           block["reference_class"], pct(o["median_schedule_ratio"]), pct(o["median_cost_ratio"]),
                           "{:.0f}%".format(o["suggested_cost_contingency_pct"]),
                           "{:.0f}%".format(o["suggested_schedule_contingency_pct"])))

    lessons = ["* [{}] {}".format(c["project_id"], c["lessons_learned"]) for c in cases[:3]]
    if lessons:
        out.append("### Lessons from the closest precedents\n" + "\n".join(lessons))

    conf = result["confidence"]
    out.append("### Confidence\n**{}** ({:.2f}). {}".format(conf["label"], conf["score"], " ".join(conf["reasons"])))
    return clean_dashes("\n\n".join(out))


def case_summary(case):
    return "{} ({}, {}): {} addressed with {}; {}, {}, final cost {}.".format(
        case["project_name"], case["industry"], case["delivery_methodology"], case["critical_issue"].lower(),
        case["intervention_strategy"].lower(), case["recovery_success"].lower(), case["outcome_status"].lower(),
        fmt_money(case["final_cost_gbp"]))


__all__ = ["SYSTEM_PROMPT", "evidence_pack", "extractive_answer", "case_summary", "ISSUES"]
