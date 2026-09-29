"""Synthetic project case generator.

Generates a large, internally consistent dataset of project cases. Numbers are
drawn with vectorised numpy operations from a causal model:

    project profile  ==>  risk pressure  ==>  critical issue and severity
    issue + chosen intervention + context  ==>  recovery  ==>  cost, schedule, quality, benefits

Because outcomes depend on the ground truth efficacy matrix in vocab.py, the
data carries a real, learnable signal about which responses work for which
problems. Text fields are composed from the numbers so every narrative agrees
with its own row.
"""

import json
import random
from datetime import date

import numpy as np
import pandas as pd

from . import narratives
from .utils import EPS, sigmoid, sub
from .vocab import (CODENAMES, CONTRACT_TYPES, EFFICACY_MAP, EFFICACY_TIERS, INDUSTRIES,
                    INTERVENTION_KEYS, INTERVENTIONS, ISSUE_KEYS, ISSUES, ORG_SIZES, OUTCOMES,
                    PHASES, RECOVERY_LEVELS, REGIONS, REGULATORY_LEVELS, SEVERITIES,
                    VOLATILITY_LEVELS, efficacy_matrix)

COLUMNS = [
    "project_id", "project_name", "industry", "project_type", "region", "delivery_methodology",
    "contract_type", "organisation_size", "start_date", "planned_end_date", "actual_end_date",
    "planned_duration_days", "actual_duration_days", "schedule_ratio", "baseline_budget_gbp",
    "final_cost_gbp", "cost_ratio", "cost_performance_index", "schedule_performance_index",
    "team_size", "stakeholder_count", "vendor_count", "dependency_count", "scope_change_requests",
    "requirements_volatility", "technical_complexity", "regulatory_exposure", "sponsor_engagement",
    "pm_experience_years", "team_turnover_pct", "risk_register_size", "risks_materialised",
    "primary_risk_category", "critical_issue", "issue_severity", "detection_phase", "root_cause",
    "intervention_strategy", "intervention_lead_time_days", "recovery_success", "outcome_status",
    "quality_defect_rate", "customer_satisfaction", "benefits_realisation_pct", "health_score",
    "problem_statement", "resolution_narrative", "lessons_learned", "tags",
]

SOFTWARE_LIKE = {"Software", "Financial Services", "Telecommunications", "Retail"}
BUILD_LIKE = {"Construction", "Energy", "Aerospace and Defence", "Logistics", "Manufacturing"}
PEOPLE_LIKE = {"Healthcare", "Public Sector", "Retail"}
RISK_CATEGORIES = ["Technical", "Commercial", "Resourcing", "Supply Chain", "Regulatory",
                   "Organisational", "Schedule", "External"]


def _choice(rng, options, weights, size):
    p = np.asarray(weights, dtype=float)
    return rng.choice(len(options), size=size, p=p / p.sum())


def _gumbel(rng, shape):
    u = rng.uniform(EPS, 1.0, size=shape)
    return np.negative(np.log(np.negative(np.log(u))))


def generate(rows=16000, seed=42):
    """Generate the dataset and return it as a pandas DataFrame."""
    rng = np.random.default_rng(seed)
    prng = random.Random(seed)
    n = int(rows)

    ind_names = list(INDUSTRIES)
    ind = _choice(rng, ind_names, [INDUSTRIES[k]["weight"] for k in ind_names], n)
    industry = np.array(ind_names, dtype=object)[ind]

    region_names = list(REGIONS)
    region = np.array(region_names, dtype=object)[_choice(rng, region_names, list(REGIONS.values()), n)]
    org_names = list(ORG_SIZES)
    org_idx = _choice(rng, org_names, [ORG_SIZES[k]["weight"] for k in org_names], n)
    org = np.array(org_names, dtype=object)[org_idx]
    org_mult = np.array([ORG_SIZES[k]["budget_multiplier"] for k in org_names])[org_idx]
    contract_names = list(CONTRACT_TYPES)
    contract = np.array(contract_names, dtype=object)[
        _choice(rng, contract_names, list(CONTRACT_TYPES.values()), n)]

    project_type = np.empty(n, dtype=object)
    methodology = np.empty(n, dtype=object)
    reg_idx = np.zeros(n, dtype=int)
    for i, name in enumerate(ind_names):
        mask = ind == i
        m = int(mask.sum())
        prof = INDUSTRIES[name]
        project_type[mask] = np.array(prof["types"], dtype=object)[rng.integers(0, len(prof["types"]), m)]
        meths = list(prof["methodologies"])
        methodology[mask] = np.array(meths, dtype=object)[
            _choice(rng, meths, list(prof["methodologies"].values()), m)]
        reg_idx[mask] = _choice(rng, REGULATORY_LEVELS, prof["regulatory"], m)

    budget_median = np.array([INDUSTRIES[k]["budget_median"] for k in ind_names], dtype=float)[ind]
    dur_median = np.array([INDUSTRIES[k]["duration_median"] for k in ind_names], dtype=float)[ind]
    cplx_mean = np.array([INDUSTRIES[k]["complexity_mean"] for k in ind_names])[ind]

    budget_noise = rng.lognormal(0.0, 0.7, n)
    budget = np.round(budget_median * org_mult * budget_noise / 1000.0) * 1000.0
    budget = np.maximum(budget, 60000.0)
    scale = budget / (budget_median * org_mult)
    complexity = np.clip(np.round(rng.normal(cplx_mean + 0.5 * np.log(scale), 1.4)), 1, 10).astype(int)
    planned_days = np.clip(np.round(dur_median * scale ** 0.3 * rng.lognormal(0.0, 0.25, n)), 45, 1800).astype(int)
    team = np.clip(np.round(3 + (budget / 1e6) ** 0.7 * 5.5 * rng.lognormal(0.0, 0.3, n)), 3, 480).astype(int)
    is_gov = (org == "Government").astype(float)
    stakeholders = rng.poisson(4 + complexity * 1.4 + is_gov * 4) + 2
    build_like = np.isin(industry, list(BUILD_LIKE)).astype(float)
    soft_like = np.isin(industry, list(SOFTWARE_LIKE)).astype(float)
    people_like = np.isin(industry, list(PEOPLE_LIKE)).astype(float)
    vendors = rng.poisson(0.5 + complexity * 0.55 + build_like * 3.0) + 1
    deps = rng.poisson(2 + complexity * 1.7) + 1

    vol_score = rng.normal(2.0, 0.9, n) + 0.22 * complexity + 0.45 * soft_like + 0.25 * people_like
    volatility = np.digitize(vol_score, [2.9, 3.9, 4.7]).astype(int)
    sponsor = (_choice(rng, [1, 2, 3, 4, 5], [8, 20, 32, 27, 13], n) + 1).astype(int)
    pm_exp = np.clip(np.round(rng.gamma(3.0, 3.2, n)), 1, 35).astype(int)
    turnover = np.round(np.clip(rng.gamma(2.0, 5.5, n), 0.5, 65.0), 1)

    fixed_price = (contract == "Fixed Price").astype(float)
    cost_plus = (contract == "Cost Plus").astype(float)
    agile = np.isin(methodology, ["Agile Scrum", "Kanban", "SAFe"]).astype(float)
    plan_driven = np.isin(methodology, ["Waterfall", "PRINCE2", "Critical Chain", "Lean Construction"]).astype(float)
    mismatch = np.clip(agile * fixed_price * (reg_idx == 2) + plan_driven * (volatility >= 2), 0, 1)

    pressure = (0.30 * complexity + 0.55 * volatility + 0.35 * np.subtract(5, sponsor) + 0.03 * turnover
                + 0.6 * mismatch + 0.25 * reg_idx + rng.normal(0.0, 0.8, n))
    pressure = np.subtract(pressure, 0.04 * pm_exp)

    small_team = (team < 12).astype(float)
    big_team = (team > 60).astype(float)
    ent_or_gov = np.isin(org, ["Enterprise", "Government"]).astype(float)
    inexperienced = (pm_exp < 5).astype(float)
    weak_sponsor = (sponsor <= 2).astype(float)
    finance_telco = np.isin(industry, ["Financial Services", "Telecommunications"]).astype(float)
    feats = {
        "scope_creep": 0.75 * volatility + 0.7 * fixed_price + 0.4 * weak_sponsor,
        "resource_contention": 0.7 * ent_or_gov + 0.025 * turnover + 0.4 * big_team,
        "critical_path_slippage": 0.12 * complexity + 0.6 * build_like + 0.02 * deps,
        "supplier_underperformance": 0.22 * vendors + 0.3 * build_like,
        "stakeholder_misalignment": 0.08 * stakeholders + 0.5 * is_gov + 0.6 * weak_sponsor,
        "requirements_ambiguity": 0.6 * volatility + 0.4 * soft_like,
        "technical_debt": 1.5 * soft_like + 0.1 * complexity,
        "integration_failure": 0.06 * deps + 0.8 * soft_like + 0.4 * finance_telco,
        "budget_overrun": 0.9 * cost_plus + 0.4 * build_like + 0.08 * complexity,
        "regulatory_delay": 1.1 * reg_idx,
        "key_person_dependency": 0.035 * turnover + 0.8 * small_team,
        "quality_defects": 0.15 * complexity + 0.4 * build_like + 0.5 * inexperienced,
        "communication_breakdown": 0.05 * stakeholders + 0.5 * big_team + 0.2 * ent_or_gov,
        "estimation_error": 0.9 * inexperienced + 0.5 * fixed_price,
        "change_resistance": 0.9 * people_like + 0.4 * is_gov,
    }
    base = {"scope_creep": 0.2, "resource_contention": 0.6, "critical_path_slippage": 0.5,
            "supplier_underperformance": 0.0, "stakeholder_misalignment": 0.1,
            "requirements_ambiguity": 0.4, "technical_debt": 0.15, "integration_failure": 0.3,
            "budget_overrun": 0.6, "regulatory_delay": 0.0, "key_person_dependency": 0.7,
            "quality_defects": 0.3, "communication_breakdown": 0.45, "estimation_error": 0.9,
            "change_resistance": 0.8}
    logits = np.stack([feats[k] + base[k] for k in ISSUE_KEYS], axis=1)
    issue_idx = np.argmax(1.1 * logits + _gumbel(rng, logits.shape), axis=1)
    issue_key = np.array(ISSUE_KEYS, dtype=object)[issue_idx]

    sev_score = pressure + rng.normal(0.0, 0.6, n)
    cuts = np.quantile(sev_score, [0.22, 0.60, 0.88])
    severity = np.digitize(sev_score, cuts).astype(int)
    phase_score = np.subtract(2.55 + rng.normal(0.0, 0.8, n), 0.03 * pm_exp) + 0.1 * severity
    phase = np.digitize(phase_score, [0.9, 1.7, 3.0, 3.9]).astype(int)

    root_cause = np.empty(n, dtype=object)
    intervention = np.empty(n, dtype=object)
    p_strong = np.clip(0.26 + 0.012 * pm_exp, 0.26, 0.62)
    u_pick = rng.uniform(0.0, 1.0, n)
    for i in range(n):
        key = issue_key[i]
        causes = ISSUES[key]["root_causes"]
        root_cause[i] = causes[prng.randrange(len(causes))]
        tiers = EFFICACY_MAP[key]
        if u_pick[i] < p_strong[i]:
            intervention[i] = prng.choice(tiers["strong"])
        elif u_pick[i] < p_strong[i] + 0.26 and tiers["moderate"]:
            intervention[i] = prng.choice(tiers["moderate"])
        elif prng.random() < 0.35:
            instinctive = ["overtime_push", "general_headcount"] + tiers["harmful"]
            intervention[i] = prng.choice(instinctive)
        else:
            intervention[i] = prng.choice(INTERVENTION_KEYS)

    eff_table = np.array(efficacy_matrix())
    int_idx = np.array([INTERVENTION_KEYS.index(k) for k in intervention])
    efficacy = eff_table[issue_idx, int_idx]
    harmful = efficacy <= EFFICACY_TIERS["harmful"] + EPS

    lead = np.round(8 + 5.5 * np.subtract(5, sponsor) + rng.exponential(12.0, n) + 3.0 * np.maximum(phase, 2)).astype(int)
    lead = np.clip(lead, 2, 180)

    pos = 4.2 * efficacy + 0.30 * sponsor + 0.035 * pm_exp + 0.72 + rng.normal(0.0, 0.35, n)
    neg = 3.28 + 0.012 * lead + 0.12 * complexity + 0.45 * severity
    p_full = sigmoid(np.subtract(pos, neg))
    p_partial = np.subtract(1.0, p_full) * (0.35 + 0.3 * efficacy)
    u_rec = rng.uniform(0.0, 1.0, n)
    recovery = np.where(u_rec < p_full, 0, np.where(u_rec < p_full + p_partial, 1, 2)).astype(int)

    residual = np.array([0.3, 1.3, 2.8])[recovery]
    base_over = 0.012 + 0.04 * severity + 0.028 * volatility + 0.006 * complexity
    sched_ratio = 1.0 + base_over * residual * rng.lognormal(0.0, 0.35, n) + 0.06 * harmful
    early = (recovery == 0) & (rng.uniform(0.0, 1.0, n) < 0.12)
    sched_ratio = np.where(early, sched_ratio * rng.uniform(0.9, 0.99, n), sched_ratio)
    budget_issue = np.isin(issue_key, ["budget_overrun", "estimation_error", "supplier_underperformance"]).astype(float)
    crash_cost = np.isin(intervention, ["crashing_specialists", "general_headcount", "overtime_push"]).astype(float)
    cost_ratio = (1.0 + (0.8 * base_over + 0.05 * budget_issue) * residual * rng.lognormal(0.0, 0.35, n)
                  + 0.35 * np.subtract(sched_ratio, 1.0) * 0.5 + 0.05 * crash_cost + 0.03 * cost_plus)
    under = (recovery == 0) & (rng.uniform(0.0, 1.0, n) < 0.15)
    cost_ratio = np.where(under, cost_ratio * rng.uniform(0.88, 0.99, n), cost_ratio)
    sched_ratio = np.clip(sched_ratio, 0.8, 3.2)
    cost_ratio = np.clip(cost_ratio, 0.8, 3.0)

    u_out = rng.uniform(0.0, 1.0, n)
    outcome = np.full(n, 1, dtype=int)
    tight = (sched_ratio <= 1.05) & (cost_ratio <= 1.05)
    late_over = (sched_ratio > 1.15) & (cost_ratio > 1.10)
    outcome = np.where(tight, 0, outcome)
    outcome = np.where(late_over, 2, outcome)
    descoped = ((intervention == "scope_rebaseline") & (u_out < 0.55)) | ((recovery == 1) & (u_out < 0.12))
    outcome = np.where(descoped, 3, outcome)
    cancelled = (recovery == 2) & (severity >= 2) & (u_out < 0.30)
    outcome = np.where(cancelled, 4, outcome)

    cancel_frac = rng.uniform(0.35, 0.85, n)
    actual_days = np.round(np.where(cancelled, planned_days * cancel_frac, planned_days * sched_ratio)).astype(int)
    actual_days = np.maximum(actual_days, 20)
    sched_ratio = np.where(cancelled, actual_days / planned_days, sched_ratio)
    cost_ratio = np.where(cancelled, cancel_frac * rng.uniform(0.9, 1.35, n), cost_ratio)
    final_cost = np.round(budget * cost_ratio / 1000.0) * 1000.0
    cost_ratio = final_cost / budget

    cpi = np.clip(1.0 / (1.0 + 0.8 * np.maximum(np.subtract(cost_ratio, 1.0), 0.0)) * rng.lognormal(0.0, 0.05, n), 0.35, 1.3)
    spi = np.clip(1.0 / (1.0 + 0.8 * np.maximum(np.subtract(sched_ratio, 1.0), 0.0)) * rng.lognormal(0.0, 0.05, n), 0.35, 1.3)

    is_scope = (issue_key == "scope_creep").astype(float)
    changes = rng.poisson(2 + 4 * volatility + 9 * is_scope + 0.3 * complexity)
    risk_size = rng.poisson(10 + 2.5 * complexity) + 3
    mat_p = np.clip(0.05 + 0.035 * pressure, 0.02, 0.8)
    risks_mat = rng.binomial(risk_size, mat_p)

    risk_cat = np.array([ISSUES[k]["risk_category"] for k in issue_key], dtype=object)
    swap = rng.uniform(0.0, 1.0, n) < 0.18
    risk_cat = np.where(swap, np.array(RISK_CATEGORIES, dtype=object)[rng.integers(0, len(RISK_CATEGORIES), n)], risk_cat)

    quality_issue = np.isin(issue_key, ["quality_defects", "technical_debt", "integration_failure"]).astype(float)
    defects = np.round(rng.gamma(2.0, 1.1, n) * (1.0 + 1.4 * quality_issue) * (0.6 + 0.5 * residual), 2)

    recovery_penalty = np.array([0.0, 0.8, 2.2])[recovery]
    csat = 8.7 + rng.normal(0.0, 0.8, n)
    csat = np.subtract(csat, 2.0 * np.maximum(np.subtract(sched_ratio, 1.0), 0.0))
    csat = np.subtract(csat, 2.2 * np.maximum(np.subtract(cost_ratio, 1.0), 0.0))
    csat = np.subtract(csat, recovery_penalty + 2.0 * cancelled)
    csat = np.clip(np.round(csat), 1, 10).astype(int)

    benefits = np.subtract(rng.normal(98.0, 11.0, n), 35.0 * np.maximum(np.subtract(sched_ratio, 1.0), 0.0))
    benefits = np.subtract(benefits, 12.0 * recovery + 15.0 * (outcome == 3))
    benefits = np.where(cancelled, rng.uniform(0.0, 15.0, n), benefits)
    benefits = np.round(np.clip(benefits, 0.0, 135.0), 1)

    health = (30.0 * (csat / 10.0) + 25.0 * np.minimum(benefits / 100.0, 1.2)
              + 20.0 / np.maximum(sched_ratio, 0.8) * 0.9 + 20.0 / np.maximum(cost_ratio, 0.8) * 0.9
              + 5.0 * np.subtract(2, recovery) / 2.0)
    health = np.where(cancelled, health * 0.5, health)
    health = np.round(np.clip(health, 1.0, 100.0), 1)

    end_limit = date(2026, 6, 30).toordinal()
    floor_start = date(2012, 1, 1).toordinal()
    window = np.maximum(np.subtract(np.subtract(end_limit, actual_days), floor_start), 1)
    start = floor_start + (rng.uniform(0.0, 1.0, n) * window).astype(int)
    planned_end = start + planned_days
    actual_end = start + actual_days

    codename = np.array(CODENAMES, dtype=object)[rng.integers(0, len(CODENAMES), n)]

    frame = pd.DataFrame({
        "project_id": ["PRJ{:06d}".format(i + 1) for i in range(n)],
        "project_name": [c + " " + t for c, t in zip(codename, project_type)],
        "industry": industry, "project_type": project_type, "region": region,
        "delivery_methodology": methodology, "contract_type": contract, "organisation_size": org,
        "start_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in start],
        "planned_end_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in planned_end],
        "actual_end_date": [date.fromordinal(int(d)).strftime("%Y/%m/%d") for d in actual_end],
        "planned_duration_days": planned_days, "actual_duration_days": actual_days,
        "schedule_ratio": np.round(sched_ratio, 3), "baseline_budget_gbp": budget.astype(np.int64),
        "final_cost_gbp": final_cost.astype(np.int64), "cost_ratio": np.round(cost_ratio, 3),
        "cost_performance_index": np.round(cpi, 3), "schedule_performance_index": np.round(spi, 3),
        "team_size": team, "stakeholder_count": stakeholders, "vendor_count": vendors,
        "dependency_count": deps, "scope_change_requests": changes,
        "requirements_volatility": np.array(VOLATILITY_LEVELS, dtype=object)[volatility],
        "technical_complexity": complexity,
        "regulatory_exposure": np.array(REGULATORY_LEVELS, dtype=object)[reg_idx],
        "sponsor_engagement": sponsor, "pm_experience_years": pm_exp, "team_turnover_pct": turnover,
        "risk_register_size": risk_size, "risks_materialised": risks_mat,
        "primary_risk_category": risk_cat,
        "critical_issue": [ISSUES[k]["name"] for k in issue_key],
        "issue_severity": np.array(SEVERITIES, dtype=object)[severity],
        "detection_phase": np.array(PHASES, dtype=object)[phase],
        "root_cause": root_cause,
        "intervention_strategy": [INTERVENTIONS[k]["name"] for k in intervention],
        "intervention_lead_time_days": lead,
        "recovery_success": np.array(RECOVERY_LEVELS, dtype=object)[recovery],
        "outcome_status": np.array(OUTCOMES, dtype=object)[outcome],
        "quality_defect_rate": defects, "customer_satisfaction": csat,
        "benefits_realisation_pct": benefits, "health_score": health,
    })

    helper = pd.DataFrame({"issue_key": issue_key, "intervention_key": intervention,
                           "volatility_idx": volatility, "severity_idx": severity})
    problems, resolutions, lessons, tag_list = [], [], [], []
    for record, extra in zip(frame.to_dict("records"), helper.to_dict("records")):
        record.update(extra)
        problems.append(narratives.problem_statement(record, prng))
        resolutions.append(narratives.resolution_narrative(record, prng))
        lessons.append(narratives.lessons_learned(record, prng))
        tag_list.append(narratives.tags(record))
    frame["problem_statement"] = problems
    frame["resolution_narrative"] = resolutions
    frame["lessons_learned"] = lessons
    frame["tags"] = tag_list
    return frame[COLUMNS]


def ground_truth():
    """The efficacy matrix as a JSON friendly structure."""
    return {
        "tiers": EFFICACY_TIERS,
        "issues": {
            ISSUES[k]["name"]: {
                tier: [INTERVENTIONS[j]["name"] for j in EFFICACY_MAP[k][tier]]
                for tier in ("strong", "moderate", "harmful")
            }
            for k in ISSUE_KEYS
        },
    }


def write_ground_truth(path):
    with open(path, "w", encoding="utf8") as fh:
        json.dump(ground_truth(), fh, indent=2)
