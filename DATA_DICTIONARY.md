# Data dictionary

The dataset holds 16,000 project cases and 49 columns. Every narrative is composed from the numbers in its own row, so text and structured fields never disagree.

<table>
<tr><th>Column</th><th>Type</th><th>Description</th><th>Example</th></tr>
<tr><td><code>project_id</code></td><td>identifier</td><td>Unique case identifier used for citations, PRJ followed by six digits.</td><td>PRJ000001</td></tr>
<tr><td><code>project_name</code></td><td>text</td><td>Codename and project type.</td><td>Saltire Point of Sale Replacement</td></tr>
<tr><td><code>industry</code></td><td>category</td><td>Sector of the delivering organisation, twelve values.</td><td>Retail</td></tr>
<tr><td><code>project_type</code></td><td>category</td><td>Specific kind of project within the industry.</td><td>Point of Sale Replacement</td></tr>
<tr><td><code>region</code></td><td>category</td><td>Geographic region of delivery.</td><td>Middle East</td></tr>
<tr><td><code>delivery_methodology</code></td><td>category</td><td>Delivery approach such as Waterfall, PRINCE2, Agile Scrum, SAFe or Hybrid.</td><td>Hybrid</td></tr>
<tr><td><code>contract_type</code></td><td>category</td><td>Commercial model such as Fixed Price, Time and Materials or Target Cost.</td><td>Internal Funding</td></tr>
<tr><td><code>organisation_size</code></td><td>category</td><td>SME, Mid Market, Enterprise or Government.</td><td>SME</td></tr>
<tr><td><code>start_date</code></td><td>date</td><td>Project start in YYYY/MM/DD format.</td><td>2019/06/28</td></tr>
<tr><td><code>planned_end_date</code></td><td>date</td><td>Baseline completion date in YYYY/MM/DD format.</td><td>2020/01/18</td></tr>
<tr><td><code>actual_end_date</code></td><td>date</td><td>Actual completion or cancellation date in YYYY/MM/DD format.</td><td>2020/04/06</td></tr>
<tr><td><code>planned_duration_days</code></td><td>integer</td><td>Baseline duration in calendar days.</td><td>204</td></tr>
<tr><td><code>actual_duration_days</code></td><td>integer</td><td>Actual duration in calendar days.</td><td>283</td></tr>
<tr><td><code>schedule_ratio</code></td><td>float</td><td>Actual duration divided by planned duration. Above 1 means late.</td><td>1.386</td></tr>
<tr><td><code>baseline_budget_gbp</code></td><td>integer</td><td>Approved baseline budget in pounds sterling.</td><td>1152000</td></tr>
<tr><td><code>final_cost_gbp</code></td><td>integer</td><td>Final outturn cost in pounds sterling.</td><td>1700000</td></tr>
<tr><td><code>cost_ratio</code></td><td>float</td><td>Final cost divided by baseline budget. Above 1 means over budget.</td><td>1.476</td></tr>
<tr><td><code>cost_performance_index</code></td><td>float</td><td>Earned value CPI measured at issue detection. Below 1 means overspending.</td><td>0.787</td></tr>
<tr><td><code>schedule_performance_index</code></td><td>float</td><td>Earned value SPI measured at issue detection. Below 1 means behind.</td><td>0.719</td></tr>
<tr><td><code>team_size</code></td><td>integer</td><td>Peak delivery team headcount.</td><td>9</td></tr>
<tr><td><code>stakeholder_count</code></td><td>integer</td><td>Number of distinct stakeholder groups.</td><td>10</td></tr>
<tr><td><code>vendor_count</code></td><td>integer</td><td>Number of suppliers and subcontractors.</td><td>3</td></tr>
<tr><td><code>dependency_count</code></td><td>integer</td><td>Number of external dependencies tracked.</td><td>11</td></tr>
<tr><td><code>scope_change_requests</code></td><td>integer</td><td>Change requests raised during delivery.</td><td>14</td></tr>
<tr><td><code>requirements_volatility</code></td><td>category</td><td>Low, Moderate, High or Severe.</td><td>High</td></tr>
<tr><td><code>technical_complexity</code></td><td>integer</td><td>Complexity rating from 1 to 10.</td><td>4</td></tr>
<tr><td><code>regulatory_exposure</code></td><td>category</td><td>Low, Medium or High.</td><td>High</td></tr>
<tr><td><code>sponsor_engagement</code></td><td>integer</td><td>Sponsor engagement rating from 1 to 5.</td><td>3</td></tr>
<tr><td><code>pm_experience_years</code></td><td>integer</td><td>Years of experience of the lead project manager.</td><td>7</td></tr>
<tr><td><code>team_turnover_pct</code></td><td>float</td><td>Percentage of the team that left during delivery.</td><td>20.4</td></tr>
<tr><td><code>risk_register_size</code></td><td>integer</td><td>Number of risks logged in the register.</td><td>24</td></tr>
<tr><td><code>risks_materialised</code></td><td>integer</td><td>Number of logged risks that became issues.</td><td>7</td></tr>
<tr><td><code>primary_risk_category</code></td><td>category</td><td>Dominant risk category, for example Technical or Commercial.</td><td>Organisational</td></tr>
<tr><td><code>critical_issue</code></td><td>category</td><td>The critical problem the case is about, fifteen values.</td><td>Change resistance</td></tr>
<tr><td><code>issue_severity</code></td><td>category</td><td>Low, Medium, High or Critical.</td><td>Medium</td></tr>
<tr><td><code>detection_phase</code></td><td>category</td><td>Lifecycle phase in which the issue was detected.</td><td>Execution</td></tr>
<tr><td><code>root_cause</code></td><td>category</td><td>Underlying cause identified in the review.</td><td>End users not involved in design</td></tr>
<tr><td><code>intervention_strategy</code></td><td>category</td><td>The response the team applied, twenty two values.</td><td>Adding general headcount to the team</td></tr>
<tr><td><code>intervention_lead_time_days</code></td><td>integer</td><td>Days from detection to the start of the response.</td><td>38</td></tr>
<tr><td><code>recovery_success</code></td><td>category</td><td>Full Recovery, Partial Recovery or Not Recovered.</td><td>Not Recovered</td></tr>
<tr><td><code>outcome_status</code></td><td>category</td><td>Final project outcome.</td><td>Delivered Late and Over Budget</td></tr>
<tr><td><code>quality_defect_rate</code></td><td>float</td><td>Defects per thousand delivered units or function points.</td><td>6.41</td></tr>
<tr><td><code>customer_satisfaction</code></td><td>integer</td><td>Client satisfaction from 1 to 10.</td><td>4</td></tr>
<tr><td><code>benefits_realisation_pct</code></td><td>float</td><td>Share of business case benefits realised, 0 to 135.</td><td>61.6</td></tr>
<tr><td><code>health_score</code></td><td>float</td><td>Composite project health from 1 to 100.</td><td>52.6</td></tr>
<tr><td><code>problem_statement</code></td><td>text</td><td>Narrative description of the problem with its symptoms and context.</td><td>The Retail project Saltire Point of Sale Replacement hit a medium s...</td></tr>
<tr><td><code>resolution_narrative</code></td><td>text</td><td>Narrative of the response and its measured result.</td><td>After 38 days the team introduced adding general headcount to the t...</td></tr>
<tr><td><code>lessons_learned</code></td><td>text</td><td>Retrospective lesson for future projects.</td><td>Adding general headcount to the team did not tackle end users not i...</td></tr>
<tr><td><code>tags</code></td><td>text</td><td>Semicolon separated keywords for filtering and search.</td><td>change resistance;end users not involved in design;adding general h...</td></tr>
</table>
