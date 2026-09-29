"""Column documentation for the Gantry AI project case dataset."""

import html

COLUMN_DOCS = {
    "project_id": ("identifier", "Unique case identifier used for citations, PRJ followed by six digits."),
    "project_name": ("text", "Codename and project type."),
    "industry": ("category", "Sector of the delivering organisation, twelve values."),
    "project_type": ("category", "Specific kind of project within the industry."),
    "region": ("category", "Geographic region of delivery."),
    "delivery_methodology": ("category", "Delivery approach such as Waterfall, PRINCE2, Agile Scrum, SAFe or Hybrid."),
    "contract_type": ("category", "Commercial model such as Fixed Price, Time and Materials or Target Cost."),
    "organisation_size": ("category", "SME, Mid Market, Enterprise or Government."),
    "start_date": ("date", "Project start in YYYY/MM/DD format."),
    "planned_end_date": ("date", "Baseline completion date in YYYY/MM/DD format."),
    "actual_end_date": ("date", "Actual completion or cancellation date in YYYY/MM/DD format."),
    "planned_duration_days": ("integer", "Baseline duration in calendar days."),
    "actual_duration_days": ("integer", "Actual duration in calendar days."),
    "schedule_ratio": ("float", "Actual duration divided by planned duration. Above 1 means late."),
    "baseline_budget_gbp": ("integer", "Approved baseline budget in pounds sterling."),
    "final_cost_gbp": ("integer", "Final outturn cost in pounds sterling."),
    "cost_ratio": ("float", "Final cost divided by baseline budget. Above 1 means over budget."),
    "cost_performance_index": ("float", "Earned value CPI measured at issue detection. Below 1 means overspending."),
    "schedule_performance_index": ("float", "Earned value SPI measured at issue detection. Below 1 means behind."),
    "team_size": ("integer", "Peak delivery team headcount."),
    "stakeholder_count": ("integer", "Number of distinct stakeholder groups."),
    "vendor_count": ("integer", "Number of suppliers and subcontractors."),
    "dependency_count": ("integer", "Number of external dependencies tracked."),
    "scope_change_requests": ("integer", "Change requests raised during delivery."),
    "requirements_volatility": ("category", "Low, Moderate, High or Severe."),
    "technical_complexity": ("integer", "Complexity rating from 1 to 10."),
    "regulatory_exposure": ("category", "Low, Medium or High."),
    "sponsor_engagement": ("integer", "Sponsor engagement rating from 1 to 5."),
    "pm_experience_years": ("integer", "Years of experience of the lead project manager."),
    "team_turnover_pct": ("float", "Percentage of the team that left during delivery."),
    "risk_register_size": ("integer", "Number of risks logged in the register."),
    "risks_materialised": ("integer", "Number of logged risks that became issues."),
    "primary_risk_category": ("category", "Dominant risk category, for example Technical or Commercial."),
    "critical_issue": ("category", "The critical problem the case is about, fifteen values."),
    "issue_severity": ("category", "Low, Medium, High or Critical."),
    "detection_phase": ("category", "Lifecycle phase in which the issue was detected."),
    "root_cause": ("category", "Underlying cause identified in the review."),
    "intervention_strategy": ("category", "The response the team applied, twenty two values."),
    "intervention_lead_time_days": ("integer", "Days from detection to the start of the response."),
    "recovery_success": ("category", "Full Recovery, Partial Recovery or Not Recovered."),
    "outcome_status": ("category", "Final project outcome."),
    "quality_defect_rate": ("float", "Defects per thousand delivered units or function points."),
    "customer_satisfaction": ("integer", "Client satisfaction from 1 to 10."),
    "benefits_realisation_pct": ("float", "Share of business case benefits realised, 0 to 135."),
    "health_score": ("float", "Composite project health from 1 to 100."),
    "problem_statement": ("text", "Narrative description of the problem with its symptoms and context."),
    "resolution_narrative": ("text", "Narrative of the response and its measured result."),
    "lessons_learned": ("text", "Retrospective lesson for future projects."),
    "tags": ("text", "Semicolon separated keywords for filtering and search."),
}


def data_dictionary_markdown(frame):
    rows = []
    for col in frame.columns:
        kind, desc = COLUMN_DOCS.get(col, ("unknown", ""))
        example = str(frame[col].iloc[0])
        if len(example) > 70:
            example = example[:67] + "..."
        rows.append("<tr><td><code>{}</code></td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            col, kind, html.escape(desc), html.escape(example)))
    header = ("# Data dictionary\n\n"
              "The dataset holds {:,} project cases and {} columns. Every narrative is composed from the "
              "numbers in its own row, so text and structured fields never disagree.\n\n").format(
        len(frame), len(frame.columns))
    table = ("<table>\n<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>\n"
             + "\n".join(rows) + "\n</table>\n")
    return header + table
