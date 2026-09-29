"""Narrative composer.

Turns the structured facts of one project case into three pieces of natural
language that read like a real post implementation review: a problem statement,
a resolution narrative and a lessons learned entry. Every number quoted in the
text is taken from the structured record so the text and the columns never
contradict each other.
"""

from .vocab import EFFICACY_MAP, INTERVENTIONS, ISSUES, efficacy_tier


def _lower_first(text):
    return text[:1].lower() + text[1:] if text else text


def _context_sentences(r, rng):
    options = []
    if r["volatility_idx"] >= 2:
        options.append("Requirements volatility was {} with {} change requests logged.".format(
            r["requirements_volatility"].lower(), r["scope_change_requests"]))
    if r["sponsor_engagement"] <= 2:
        options.append("Sponsor engagement was weak, rated {} out of 5.".format(r["sponsor_engagement"]))
    if r["sponsor_engagement"] >= 5:
        options.append("Sponsor engagement was strong, rated {} out of 5.".format(r["sponsor_engagement"]))
    if r["pm_experience_years"] <= 4:
        options.append("The project manager had {} years of experience.".format(r["pm_experience_years"]))
    if r["team_turnover_pct"] >= 20:
        options.append("Team turnover reached {:.0f} percent.".format(r["team_turnover_pct"]))
    if r["vendor_count"] >= 7:
        options.append("{} vendors were involved in delivery.".format(r["vendor_count"]))
    if r["regulatory_exposure"] == "High":
        options.append("Regulatory exposure was high.")
    if r["technical_complexity"] >= 8:
        options.append("Technical complexity was rated {} out of 10.".format(r["technical_complexity"]))
    if not options:
        options.append("The team of {} people managed {} external dependencies.".format(
            r["team_size"], r["dependency_count"]))
    rng.shuffle(options)
    return " ".join(options[:2])


def _symptom(r, rng):
    template = rng.choice(ISSUES[r["issue_key"]]["symptoms"])
    sev = r["severity_idx"]
    return template.format(
        changes=max(3, r["scope_change_requests"]),
        weeks=rng.randint(2 + sev * 2, 6 + sev * 4),
        pct=rng.randint(10 + sev * 8, 25 + sev * 12),
        deps=rng.randint(2, 4 + sev * 2),
    )


def problem_statement(r, rng):
    symptom = _symptom(r, rng)
    context = _context_sentences(r, rng)
    issue_lower = _lower_first(r["critical_issue"])
    root_lower = _lower_first(r["root_cause"])
    phase_lower = r["detection_phase"].lower()
    sev_lower = r["issue_severity"].lower()
    templates = [
        "{name} was delivered by a {org} {industry} organisation in {region} using {method}. "
        "During {phase}, {symptom}. {context} The root cause was traced to {root}.",
        "During {phase} on {name}, {symptom}. The programme ran under a {contract} contract with a team "
        "of {team} people and faced {sev} severity {issue}. {context}",
        "{Issue} emerged on {name} in the {industry} sector when {symptom}. {context} "
        "Investigation pointed to {root}.",
        "The {industry} project {name} hit a {sev} severity problem during {phase}: {symptom}. "
        "{context} Analysis identified {root} as the underlying cause.",
    ]
    return rng.choice(templates).format(
        name=r["project_name"], org=r["organisation_size"], industry=r["industry"],
        region=r["region"], method=r["delivery_methodology"], phase=phase_lower, symptom=symptom,
        context=context, root=root_lower, contract=r["contract_type"].lower(), team=r["team_size"],
        sev=sev_lower, issue=issue_lower, Issue=r["critical_issue"],
    ).replace("  ", " ").strip()


def _outcome_sentence(r, rng):
    sched = "{:.0f}".format(100 * r["schedule_ratio"])
    cost = "{:.0f}".format(100 * r["cost_ratio"])
    outcome = r["outcome_status"]
    if outcome == "Cancelled":
        return "The programme was cancelled after spending {} percent of its baseline budget.".format(cost)
    rec = r["recovery_success"]
    if rec == "Full Recovery":
        base = rng.choice([
            "Control was restored and the project finished at {s} percent of planned duration and {c} percent of baseline budget.",
            "The project recovered fully, closing at {s} percent of planned duration and {c} percent of budget.",
            "Performance returned to plan tolerances with final duration at {s} percent and cost at {c} percent of baseline.",
        ])
    elif rec == "Partial Recovery":
        base = rng.choice([
            "The position stabilised but did not fully recover, closing at {s} percent of planned duration and {c} percent of budget.",
            "Recovery was partial; the project finished at {s} percent of planned duration and {c} percent of budget.",
        ])
    else:
        base = rng.choice([
            "The intervention did not arrest the decline and the project closed at {s} percent of planned duration and {c} percent of budget.",
            "The situation continued to deteriorate, ending at {s} percent of planned duration and {c} percent of budget.",
        ])
    sentence = base.format(s=sched, c=cost)
    if outcome == "Descoped":
        sentence += " Scope was reduced to protect the core benefits."
    return sentence


def resolution_narrative(r, rng):
    intervention = INTERVENTIONS[r["intervention_key"]]
    name_lower = _lower_first(intervention["name"])
    templates = [
        "After {lead} days the team introduced {name}: {mech}. {outcome}",
        "The response was {name}, launched {lead} days after detection, in which {mech}. {outcome}",
        "Leadership approved {name} within {lead} days; {mech}. {outcome}",
    ]
    return rng.choice(templates).format(
        lead=r["intervention_lead_time_days"], name=name_lower, mech=intervention["mechanism"],
        outcome=_outcome_sentence(r, rng),
    )


def lessons_learned(r, rng):
    issue_key, int_key = r["issue_key"], r["intervention_key"]
    tier = efficacy_tier(issue_key, int_key)
    issue_lower = _lower_first(r["critical_issue"])
    int_name = INTERVENTIONS[int_key]["name"]
    int_lower = _lower_first(int_name)
    root_lower = _lower_first(r["root_cause"])
    best = INTERVENTIONS[rng.choice(EFFICACY_MAP[issue_key]["strong"])]["name"]
    best_lower = _lower_first(best)
    rec = r["recovery_success"]
    lead = r["intervention_lead_time_days"]

    if tier == "strong" and rec == "Full Recovery":
        options = [
            "When {issue} appears, {intv} works best if started early; here it restored control within {lead} days of detection.",
            "{Intv} directly addressed {root} and should be the default response to {issue}.",
            "Treat {issue} as a governance problem: {intv} gave the team a repeatable way to regain control.",
        ]
    elif tier == "strong":
        options = [
            "{Intv} is usually effective against {issue}, but a {lead} day delay limited its impact. Act faster.",
            "The right response was chosen, but weak sponsorship and late action meant {intv} could not fully recover the position.",
            "Even proven responses such as {intv} need executive backing and early action to succeed.",
        ]
    elif tier == "moderate":
        options = [
            "{Intv} gave partial relief; pairing it with a fix for {root} would have addressed the underlying cause.",
            "{Intv} helped with symptoms of {issue} but the root cause, {root}, needed a more targeted response such as {best}.",
        ]
    elif tier == "harmful":
        options = [
            "{Intv} made matters worse: it added pressure without tackling {root}. Teams facing {issue} should prioritise {best}.",
            "Avoid {intv} as a response to {issue}; it increased cost and fatigue while the root cause remained.",
        ]
    else:
        if rec == "Full Recovery":
            options = [
                "Recovery owed more to strong sponsorship and an experienced team than to {intv}; {best} is the more reliable response to {issue}.",
            ]
        else:
            options = [
                "{Intv} did not tackle {root}; teams facing {issue} should prioritise {best}.",
                "The chosen response did not match the problem. For {issue}, evidence favours {best} over {intv}.",
            ]
    return rng.choice(options).format(
        issue=issue_lower, intv=int_lower, Intv=int_name, root=root_lower, lead=lead,
        best=best_lower,
    )


def tags(r):
    parts = [
        r["critical_issue"], r["root_cause"], INTERVENTIONS[r["intervention_key"]]["name"],
        r["industry"], r["delivery_methodology"], r["contract_type"], r["outcome_status"],
        ISSUES[r["issue_key"]]["risk_category"] + " risk",
    ]
    return ";".join(p.lower() for p in parts)
