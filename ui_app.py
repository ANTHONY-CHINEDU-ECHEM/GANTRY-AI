"""Gantry AI web interface. Start with: python gantry.py ui"""

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gantry_ai.engine import GantryEngine  # noqa: E402
from gantry_ai.utils import fmt_money  # noqa: E402

st.set_page_config(page_title="Gantry AI", layout="wide")

EXAMPLES = [
    "Our main supplier on a fixed price construction project keeps missing delivery dates and the steering group is losing patience. What should we do?",
    "Our agile banking platform team is weeks behind, the only architect who understands the core is resigning and velocity keeps dropping.",
    "Costs are spiralling on our offshore wind substation and contingency is almost gone.",
    "Frontline staff in the hospital are refusing to use the new patient record system.",
    "Stakeholders keep adding new features to our retail loyalty programme and the baseline keeps expanding.",
]


@st.cache_resource(show_spinner="Loading the Gantry index")
def engine():
    return GantryEngine.shared()


def pct(x):
    return "{:.0%}".format(x)


def page_ask(eng):
    st.header("Ask Gantry")
    st.caption("Describe a delivery problem in your own words. Gantry diagnoses it, finds comparable "
               "historical cases and ranks responses by how often they actually worked.")
    example = st.selectbox("Start from an example or write your own", ["Write my own"] + EXAMPLES)
    question = st.text_area("Your question", value="" if example == "Write my own" else example, height=110)
    opts = eng.options()
    with st.expander("Filters and generation settings"):
        c1, c2, c3 = st.columns(3)
        industries = c1.multiselect("Industry", opts["industries"])
        methods = c2.multiselect("Methodology", opts["methodologies"])
        contracts = c3.multiselect("Contract", opts["contracts"])
        c4, c5 = st.columns(2)
        provider = c4.selectbox("Answer provider", ["extractive", "anthropic", "ollama"])
        k = c5.slider("Precedent cases in context", 3, 15, 8)
    if not st.button("Ask Gantry", type="primary") or not question.strip():
        return
    filters = {}
    if industries:
        filters["industry"] = industries
    if methods:
        filters["delivery_methodology"] = methods
    if contracts:
        filters["contract_type"] = contracts
    result = eng.ask(question, filters=filters or None, k=k, provider=provider)

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Diagnosis", result["diagnosis"][0]["issue"] if result["diagnosis"] else "Unclear")
    m2.metric("Confidence", "{} ({:.2f})".format(result["confidence"]["label"], result["confidence"]["score"]))
    m3.metric("Grounding score", "{:.2f}".format(result["verification"]["grounding_score"]))
    m4.metric("Answer time", "{:.1f} ms".format(result["timings_ms"]["total"]))
    for note in result["notes"]:
        st.info(note)
    st.markdown(result["answer"])

    tab_e, tab_c, tab_v = st.tabs(["Evidence", "Precedent cases", "Verification"])
    with tab_e:
        for block in result["recommendations"]:
            st.subheader(block["issue"])
            st.caption("Reference class: {}. Baseline full recovery {}.".format(
                block["reference_class"], pct(block["baseline_full_recovery"])))
            table = pd.DataFrame(block["all_interventions"])
            if not table.empty:
                table = table[["intervention", "cases", "full_recovery_rate", "confidence_lower_bound",
                               "median_cost_ratio", "median_schedule_ratio", "median_lead_time_days"]]
                st.dataframe(table, hide_index=True, width="stretch")
                st.bar_chart(table.set_index("intervention")["confidence_lower_bound"])
    with tab_c:
        for case in result["cases"]:
            title = "{}  {}  ({})".format(case["project_id"], case["project_name"], case["recovery_success"])
            with st.expander(title):
                st.write("**Problem.** " + case["problem_statement"])
                st.write("**Resolution.** " + case["resolution_narrative"])
                st.write("**Lesson.** " + case["lessons_learned"])
                st.caption("{} | {} | {} | budget {} | duration {:.0%} of plan | cost {:.0%} of budget".format(
                    case["industry"], case["delivery_methodology"], case["contract_type"],
                    fmt_money(case["baseline_budget_gbp"]), case["schedule_ratio"], case["cost_ratio"]))
    with tab_v:
        st.json(result["verification"])
        st.json(result["timings_ms"])


def page_search(eng):
    st.header("Case explorer")
    query = st.text_input("Search the case base", "vendor missed delivery dates on a fibre rollout")
    c1, c2 = st.columns(2)
    mode = c1.selectbox("Retrieval method", ["hybrid", "fusion", "bm25", "dense"])
    k = c2.slider("Results", 5, 50, 15)
    if not query.strip():
        return
    out = eng.search(query, k=k, mode=mode)
    frame = pd.DataFrame(out["results"])
    st.caption("Retrieved in {} ms".format(out["timings_ms"]["retrieve"]))
    st.dataframe(frame[["project_id", "score", "industry", "critical_issue", "intervention_strategy",
                        "recovery_success", "outcome_status", "cost_ratio", "schedule_ratio"]],
                 hide_index=True, width="stretch")
    pick = st.selectbox("Open a case", frame["project_id"].tolist())
    case = eng.case(pick)
    st.write("**Problem.** " + case["problem_statement"])
    st.write("**Resolution.** " + case["resolution_narrative"])
    st.write("**Lesson.** " + case["lessons_learned"])


def page_assess(eng):
    st.header("Risk assessor")
    st.caption("Profile a project before or during delivery to see its predicted exposure, the drivers "
               "behind it and a prepared response for each likely issue.")
    opts = eng.options()
    with st.form("profile"):
        c1, c2, c3 = st.columns(3)
        profile = {
            "industry": c1.selectbox("Industry", opts["industries"]),
            "delivery_methodology": c2.selectbox("Methodology", opts["methodologies"]),
            "contract_type": c3.selectbox("Contract", opts["contracts"]),
            "organisation_size": c1.selectbox("Organisation size", opts["organisation_sizes"]),
            "region": c2.selectbox("Region", opts["regions"]),
            "requirements_volatility": c3.selectbox("Requirements volatility", opts["volatility"], index=1),
            "regulatory_exposure": c1.selectbox("Regulatory exposure", opts["regulatory"], index=1),
            "technical_complexity": c2.slider("Technical complexity", 1, 10, 6),
            "sponsor_engagement": c3.slider("Sponsor engagement", 1, 5, 3),
            "pm_experience_years": c1.slider("Project manager experience in years", 1, 35, 8),
            "team_size": c2.number_input("Team size", 1, 1000, 25),
            "vendor_count": c3.number_input("Vendors", 0, 100, 4),
            "dependency_count": c1.number_input("External dependencies", 0, 200, 12),
            "stakeholder_count": c2.number_input("Stakeholder groups", 0, 200, 12),
            "team_turnover_pct": c3.slider("Expected team turnover percent", 0, 60, 10),
            "baseline_budget_gbp": c1.number_input("Baseline budget in GBP", 10000, 2_000_000_000, 2_500_000, step=100000),
            "planned_duration_days": c2.number_input("Planned duration in days", 30, 3000, 300),
        }
        submitted = st.form_submit_button("Assess risk", type="primary")
    if not submitted:
        return
    out = eng.assess(profile)
    cols = st.columns(len(out["risks"]))
    for col, risk in zip(cols, out["risks"].values()):
        col.metric(risk["label"], pct(risk["probability"]), help="Portfolio base rate {}".format(pct(risk["base_rate"])))
        for d in risk["drivers"]:
            col.caption("{}: {}".format(d["factor"], d["effect"]))
    st.subheader("Prepared playbook")
    st.dataframe(pd.DataFrame(out["playbook"]), hide_index=True, width="stretch")


def page_insights(eng):
    st.header("Portfolio insights")
    s = eng.stats()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Cases", "{:,}".format(s["cases"]))
    c2.metric("Portfolio value", fmt_money(s["total_portfolio_value_gbp"]))
    c3.metric("Median cost ratio", "{:.2f}".format(s["median_cost_ratio"]))
    c4.metric("Median schedule ratio", "{:.2f}".format(s["median_schedule_ratio"]))
    a, b = st.columns(2)
    a.subheader("Critical issues")
    a.bar_chart(pd.Series(s["issues"]))
    b.subheader("Outcomes")
    b.bar_chart(pd.Series(s["outcomes"]))
    st.subheader("What works for each issue")
    opts = eng.options()
    issue = st.selectbox("Issue", opts["issues"])
    industry = st.selectbox("Industry context", ["All"] + opts["industries"])
    block = eng.recommend(issue, industry=None if industry == "All" else industry)
    st.caption("Reference class: {}".format(block["reference_class"]))
    table = pd.DataFrame(block["all_interventions"]).set_index("intervention")
    st.bar_chart(table["full_recovery_rate"])
    st.dataframe(table, width="stretch")


def main():
    eng = engine()
    st.sidebar.title("Gantry AI")
    st.sidebar.caption("Evidence grounded intelligence for complex project delivery.")
    page = st.sidebar.radio("Workspace", ["Ask Gantry", "Case explorer", "Risk assessor", "Portfolio insights"])
    st.sidebar.divider()
    st.sidebar.caption("{:,} historical cases indexed".format(eng.index.size))
    {"Ask Gantry": page_ask, "Case explorer": page_search, "Risk assessor": page_assess,
     "Portfolio insights": page_insights}[page](eng)


main()
