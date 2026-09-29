"""Domain knowledge for Gantry AI.

This module is the single source of truth for the project management world that
the synthetic dataset simulates and that the retrieval engine understands. It
defines industries, delivery contexts, critical issues, interventions and the
ground truth efficacy matrix that links issues to the interventions that
genuinely resolve them.

The efficacy matrix is what makes the dataset scientifically useful. Outcomes in
the generated data are driven by it (together with sponsor engagement, project
manager experience, response speed, complexity and severity), so a retrieval
system that recommends interventions from evidence can be scored against a
known truth rather than against opinion.
"""

INDUSTRIES = {
    "Construction": {
        "weight": 11, "budget_median": 18_000_000, "duration_median": 540, "complexity_mean": 6.0,
        "regulatory": (0.25, 0.50, 0.25),
        "methodologies": {"Waterfall": 30, "PRINCE2": 25, "Lean Construction": 20, "Critical Chain": 10, "Hybrid": 15},
        "types": ["Commercial Office Build", "Hospital Wing Extension", "Rail Station Upgrade",
                  "Bridge Refurbishment", "Residential Development", "School Campus Build"],
    },
    "Software": {
        "weight": 14, "budget_median": 2_400_000, "duration_median": 300, "complexity_mean": 6.2,
        "regulatory": (0.60, 0.30, 0.10),
        "methodologies": {"Agile Scrum": 35, "Kanban": 12, "SAFe": 15, "Hybrid": 25, "Waterfall": 13},
        "types": ["SaaS Platform Rebuild", "Mobile Banking App", "Data Platform Migration",
                  "ERP Implementation", "Cloud Migration", "Customer Portal Launch"],
    },
    "Healthcare": {
        "weight": 8, "budget_median": 6_500_000, "duration_median": 420, "complexity_mean": 6.6,
        "regulatory": (0.05, 0.35, 0.60),
        "methodologies": {"PRINCE2": 30, "Hybrid": 30, "Waterfall": 20, "Agile Scrum": 20},
        "types": ["Electronic Patient Record Rollout", "Clinical Pathway Redesign",
                  "Diagnostic Imaging Upgrade", "Telehealth Service Launch", "Pharmacy Automation"],
    },
    "Financial Services": {
        "weight": 10, "budget_median": 9_000_000, "duration_median": 390, "complexity_mean": 6.8,
        "regulatory": (0.05, 0.30, 0.65),
        "methodologies": {"SAFe": 25, "Hybrid": 30, "Agile Scrum": 20, "Waterfall": 15, "PRINCE2": 10},
        "types": ["Core Banking Replacement", "Regulatory Reporting Programme", "Payments Modernisation",
                  "Fraud Analytics Platform", "Open Banking Integration"],
    },
    "Energy": {
        "weight": 7, "budget_median": 42_000_000, "duration_median": 720, "complexity_mean": 7.0,
        "regulatory": (0.10, 0.40, 0.50),
        "methodologies": {"Waterfall": 35, "PRINCE2": 20, "Critical Chain": 20, "Hybrid": 25},
        "types": ["Offshore Wind Substation", "Grid Reinforcement", "Smart Meter Rollout",
                  "Solar Farm Construction", "Battery Storage Facility"],
    },
    "Manufacturing": {
        "weight": 8, "budget_median": 7_500_000, "duration_median": 360, "complexity_mean": 5.8,
        "regulatory": (0.35, 0.45, 0.20),
        "methodologies": {"Waterfall": 25, "Hybrid": 25, "Critical Chain": 20, "Kanban": 15, "PRINCE2": 15},
        "types": ["Production Line Automation", "Factory Relocation", "MES Deployment",
                  "Lean Transformation", "Robotic Cell Integration"],
    },
    "Public Sector": {
        "weight": 9, "budget_median": 5_500_000, "duration_median": 450, "complexity_mean": 6.1,
        "regulatory": (0.15, 0.45, 0.40),
        "methodologies": {"PRINCE2": 35, "Agile Scrum": 20, "Hybrid": 25, "Waterfall": 20},
        "types": ["Digital Service Transformation", "Shared Services Consolidation",
                  "Case Management System", "Estates Rationalisation", "Citizen Identity Platform"],
    },
    "Telecommunications": {
        "weight": 7, "budget_median": 12_000_000, "duration_median": 420, "complexity_mean": 6.5,
        "regulatory": (0.25, 0.50, 0.25),
        "methodologies": {"Hybrid": 30, "SAFe": 20, "Waterfall": 25, "Agile Scrum": 15, "Kanban": 10},
        "types": ["Fibre Network Rollout", "5G Site Deployment", "Billing System Replacement",
                  "Network Operations Centre Upgrade"],
    },
    "Retail": {
        "weight": 7, "budget_median": 3_200_000, "duration_median": 270, "complexity_mean": 5.4,
        "regulatory": (0.60, 0.32, 0.08),
        "methodologies": {"Agile Scrum": 30, "Kanban": 20, "Hybrid": 30, "Waterfall": 20},
        "types": ["Omnichannel Commerce Platform", "Store Refit Programme", "Supply Chain Visibility Platform",
                  "Loyalty Programme Launch", "Point of Sale Replacement"],
    },
    "Pharmaceuticals": {
        "weight": 6, "budget_median": 11_000_000, "duration_median": 480, "complexity_mean": 7.2,
        "regulatory": (0.02, 0.18, 0.80),
        "methodologies": {"Waterfall": 35, "PRINCE2": 20, "Hybrid": 35, "Agile Scrum": 10},
        "types": ["Clinical Trial Management System", "Manufacturing Site Validation",
                  "Serialisation Compliance", "Laboratory Automation"],
    },
    "Aerospace and Defence": {
        "weight": 5, "budget_median": 35_000_000, "duration_median": 780, "complexity_mean": 7.8,
        "regulatory": (0.05, 0.30, 0.65),
        "methodologies": {"Waterfall": 40, "Hybrid": 25, "Critical Chain": 20, "PRINCE2": 15},
        "types": ["Avionics Integration", "MRO Facility Upgrade", "Simulator Development",
                  "Secure Communications Programme"],
    },
    "Logistics": {
        "weight": 8, "budget_median": 6_000_000, "duration_median": 330, "complexity_mean": 5.9,
        "regulatory": (0.50, 0.40, 0.10),
        "methodologies": {"Hybrid": 30, "Waterfall": 25, "Kanban": 20, "Agile Scrum": 15, "Critical Chain": 10},
        "types": ["Warehouse Automation", "Fleet Telematics Rollout", "Transport Management System",
                  "Distribution Centre Build"],
    },
}

REGIONS = {
    "UK and Ireland": 26, "Western Europe": 16, "Nordics": 6, "North America": 18,
    "Middle East": 8, "Asia Pacific": 12, "Sub Saharan Africa": 7, "Latin America": 7,
}

ORG_SIZES = {
    "SME": {"weight": 18, "budget_multiplier": 0.35},
    "Mid Market": {"weight": 30, "budget_multiplier": 0.75},
    "Enterprise": {"weight": 37, "budget_multiplier": 1.40},
    "Government": {"weight": 15, "budget_multiplier": 1.10},
}

CONTRACT_TYPES = {
    "Fixed Price": 28, "Time and Materials": 22, "Cost Plus": 10,
    "Target Cost": 12, "Alliance": 6, "Internal Funding": 22,
}

METHODOLOGIES = ["Waterfall", "PRINCE2", "Agile Scrum", "Kanban", "SAFe", "Hybrid",
                 "Critical Chain", "Lean Construction"]

PHASES = ["Initiation", "Planning", "Execution", "Monitoring and Control", "Closure"]
SEVERITIES = ["Low", "Medium", "High", "Critical"]
VOLATILITY_LEVELS = ["Low", "Moderate", "High", "Severe"]
REGULATORY_LEVELS = ["Low", "Medium", "High"]
RECOVERY_LEVELS = ["Full Recovery", "Partial Recovery", "Not Recovered"]
OUTCOMES = ["Delivered On Target", "Delivered With Variance", "Delivered Late and Over Budget",
            "Descoped", "Cancelled"]

CODENAMES = [
    "Northgate", "Kestrel", "Meridian", "Ashford", "Halcyon", "Ironbridge", "Lumen", "Solway",
    "Pennine", "Harbourline", "Atlas", "Blackwater", "Cobalt", "Driftwood", "Elmstead", "Falcon",
    "Granite", "Heron", "Ivory", "Juniper", "Kingsway", "Larch", "Marlow", "Nimbus", "Oakridge",
    "Pioneer", "Quarry", "Redwing", "Saltire", "Thornbury", "Upland", "Vantage", "Westmoor",
    "Yarrow", "Zenith", "Bramble", "Cedar", "Dunmore", "Evergreen", "Foxglove", "Greystone",
    "Highfield", "Islay", "Jura", "Keystone", "Longford", "Millbank", "Newhaven", "Orion",
    "Portland", "Riverside", "Sterling", "Tamar", "Ullswater", "Victoria", "Wharfedale", "Arden",
    "Beacon", "Clyde", "Derwent", "Exmoor", "Fenwick", "Galloway", "Hadrian", "Irwell", "Kielder",
]

# Critical issues. Each issue carries the language the dataset uses to narrate it
# (symptoms), independent phrasing used only by the evaluation harness (queries)
# and the keywords the query understanding layer listens for (keywords).
ISSUES = {
    "scope_creep": {
        "name": "Scope creep",
        "risk_category": "Commercial",
        "root_causes": ["Weak change control at project start", "Ambiguous contract scope boundaries",
                        "Sponsor accepting informal requests", "Gold plating by the delivery team"],
        "symptoms": [
            "{changes} change requests arrived in {weeks} weeks and roughly {pct} percent were approved without impact analysis",
            "the backlog grew by {pct} percent while the delivery date stayed fixed",
            "informal feature requests from business units expanded the baseline scope by {pct} percent",
        ],
        "queries": [
            "stakeholders keep adding new features and the baseline keeps expanding",
            "requirements keep growing after sign off and nobody assesses the impact of new asks",
            "the business keeps asking for extras and our scope is out of control",
        ],
        "keywords": ["scope creep", "scope", "new features", "extra features", "additional requirements",
                     "change requests", "gold plating", "baseline keeps", "keeps adding", "out of control"],
    },
    "resource_contention": {
        "name": "Resource contention",
        "risk_category": "Resourcing",
        "root_causes": ["Shared specialists across competing programmes", "No portfolio level capacity planning",
                        "Line managers reclaiming seconded staff", "Unplanned business as usual demand"],
        "symptoms": [
            "{pct} percent of planned specialist hours were lost to competing programmes",
            "key engineers were split across {deps} concurrent initiatives",
            "the resource plan was {pct} percent under capacity for three consecutive months",
        ],
        "queries": [
            "our best people are being pulled onto other programmes and we cannot staff the work",
            "we share specialists with competing projects and never get enough capacity",
            "staff are double booked across initiatives and tasks sit idle",
        ],
        "keywords": ["resource", "resources", "capacity", "double booked", "pulled onto", "staffing",
                     "not enough people", "shared specialists", "availability"],
    },
    "critical_path_slippage": {
        "name": "Critical path slippage",
        "risk_category": "Schedule",
        "root_causes": ["Optimistic activity durations", "Unmanaged dependencies on the critical path",
                        "Late long lead procurement", "Sequential work that could run in parallel"],
        "symptoms": [
            "the critical path slipped {weeks} weeks and float on near critical activities was exhausted",
            "milestone delivery fell {pct} percent behind the baseline schedule",
            "{deps} dependent activities on the critical path finished late in the same quarter",
        ],
        "queries": [
            "we are weeks behind schedule and the go live date is at risk",
            "milestones keep slipping and there is no float left in the plan",
            "the project is running late and the programme cannot move the deadline",
        ],
        "keywords": ["behind schedule", "weeks behind", "months behind", "behind plan", "slipping", "slippage",
                     "running late", "late", "delay", "delayed", "overdue", "deadline", "critical path",
                     "milestone", "float", "go live date"],
    },
    "supplier_underperformance": {
        "name": "Supplier underperformance",
        "risk_category": "Supply Chain",
        "root_causes": ["Supplier capacity overstated at tender", "Weak performance clauses in the contract",
                        "Single source dependency on a critical component", "Subcontractor insolvency risk ignored"],
        "symptoms": [
            "the principal supplier missed {deps} consecutive delivery commitments",
            "vendor deliverables failed acceptance {changes} times in one quarter",
            "a critical subcontractor delivered {pct} percent of contracted output",
        ],
        "queries": [
            "our vendor keeps missing delivery dates and quality of their work is poor",
            "the subcontractor is failing to deliver what the contract promised",
            "a key supplier is late with components and we depend on them entirely",
        ],
        "keywords": ["supplier", "vendor", "subcontractor", "contractor", "third party", "outsourced",
                     "procurement", "components", "partner is failing"],
    },
    "stakeholder_misalignment": {
        "name": "Stakeholder misalignment",
        "risk_category": "Organisational",
        "root_causes": ["Conflicting success measures between sponsors", "Unclear decision rights",
                        "Sponsor changed mid programme", "Business units never agreed the target state"],
        "symptoms": [
            "{deps} senior stakeholders held conflicting views on the target outcome",
            "steering decisions were reversed {changes} times in two quarters",
            "the business case was challenged by {deps} directorates after approval",
        ],
        "queries": [
            "executives disagree on priorities and decisions keep getting reversed",
            "our sponsors want different outcomes and the steering group cannot agree",
            "senior leaders are not aligned on what success looks like",
        ],
        "keywords": ["stakeholder", "stakeholders", "sponsor", "sponsors", "executives", "leadership",
                     "not aligned", "disagree", "politics", "steering", "priorities"],
    },
    "requirements_ambiguity": {
        "name": "Requirements ambiguity",
        "risk_category": "Technical",
        "root_causes": ["Requirements captured without user research", "Acceptance criteria missing",
                        "Business rules held in the heads of a few experts", "Rushed discovery phase"],
        "symptoms": [
            "{pct} percent of user stories lacked testable acceptance criteria",
            "delivered features were rejected in user acceptance testing {changes} times",
            "the specification contained {changes} unresolved open questions at build start",
        ],
        "queries": [
            "nobody can tell us exactly what the users need and specs are vague",
            "acceptance testing keeps failing because the requirements were unclear",
            "the specification is full of open questions and assumptions",
        ],
        "keywords": ["requirements", "unclear", "vague", "ambiguous", "ambiguity", "acceptance criteria",
                     "specification", "specs", "what the users need", "user stories"],
    },
    "technical_debt": {
        "name": "Technical debt accumulation",
        "risk_category": "Technical",
        "root_causes": ["Shortcuts taken to hit early milestones", "Legacy platform constraints",
                        "No automated test coverage", "Architecture decisions deferred"],
        "symptoms": [
            "velocity fell {pct} percent over {weeks} weeks as rework consumed capacity",
            "automated test coverage stood at {pct} percent on core services",
            "{changes} production incidents traced back to deferred refactoring",
        ],
        "queries": [
            "the codebase is fragile and every change breaks something else",
            "velocity keeps dropping because the team spends its time on rework of legacy code",
            "shortcuts from earlier releases are now slowing every sprint",
        ],
        "keywords": ["technical debt", "legacy", "codebase", "refactor", "refactoring", "fragile",
                     "rework", "velocity", "architecture", "shortcuts"],
    },
    "integration_failure": {
        "name": "Integration failure",
        "risk_category": "Technical",
        "root_causes": ["Interfaces specified late", "No shared integration environment",
                        "Third party APIs changed without notice", "Data mapping underestimated"],
        "symptoms": [
            "{changes} interfaces failed end to end testing in the first integration cycle",
            "data reconciliation between systems showed {pct} percent mismatched records",
            "the integration phase overran by {weeks} weeks",
        ],
        "queries": [
            "systems will not talk to each other and interface testing keeps failing",
            "data does not reconcile between the old and new platforms",
            "our APIs break whenever the other team deploys",
        ],
        "keywords": ["integration", "interface", "interfaces", "api", "apis", "data mapping",
                     "reconcile", "migration", "systems will not", "end to end"],
    },
    "budget_overrun": {
        "name": "Budget overrun",
        "risk_category": "Commercial",
        "root_causes": ["Contingency consumed by early risks", "Inflation in materials and labour",
                        "Cost baseline built on outdated rates", "Uncontrolled variation orders"],
        "symptoms": [
            "forecast outturn exceeded the approved budget by {pct} percent",
            "contingency was fully drawn down by {pct} percent completion",
            "monthly burn ran {pct} percent above plan for a full quarter",
        ],
        "queries": [
            "we are overspending and the forecast is way above the approved budget",
            "costs are spiralling and contingency is almost gone",
            "the money is running out before the work is finished",
        ],
        "keywords": ["budget", "over budget", "overspend", "overspending", "cost", "costs", "money",
                     "contingency", "spiralling", "burn rate", "forecast outturn", "funding"],
    },
    "regulatory_delay": {
        "name": "Regulatory approval delay",
        "risk_category": "Regulatory",
        "root_causes": ["Regulator engaged too late", "Submission dossier incomplete",
                        "Changing regulatory guidance", "Planning consent conditions underestimated"],
        "symptoms": [
            "regulatory approval arrived {weeks} weeks later than planned",
            "the regulator issued {changes} rounds of clarification questions",
            "consent conditions added {pct} percent to remaining scope",
        ],
        "queries": [
            "we are stuck waiting for the regulator to approve our submission",
            "compliance approval is holding up the whole programme",
            "planning consent conditions are delaying construction",
        ],
        "keywords": ["regulator", "regulatory", "approval", "compliance", "consent", "planning permission",
                     "submission", "licence", "audit", "certification"],
    },
    "key_person_dependency": {
        "name": "Key person dependency",
        "risk_category": "Resourcing",
        "root_causes": ["Critical knowledge held by one expert", "No succession planning",
                        "Contractor holding undocumented design knowledge", "Burnout of a lead engineer"],
        "symptoms": [
            "a single architect approved {pct} percent of design decisions",
            "the departure of a lead engineer stalled {deps} workstreams",
            "{deps} critical tasks could only be performed by one contractor",
        ],
        "queries": [
            "only one person knows how the system works and they are about to leave",
            "we depend on a single expert and everything waits for them",
            "our lead engineer resigned and nobody else understands the design",
        ],
        "keywords": ["one person", "single expert", "key person", "resigned", "resign", "quit", "leaving",
                     "leave", "knowledge", "bus factor", "succession", "only one", "burnout", "only architect",
                     "lead engineer", "single point of failure"],
    },
    "quality_defects": {
        "name": "Quality defects escalation",
        "risk_category": "Technical",
        "root_causes": ["Inspection and test activity compressed", "Unclear quality standards",
                        "Inexperienced workforce on critical tasks", "Defect root causes never analysed"],
        "symptoms": [
            "open defects rose to {changes} with {pct} percent rated severe",
            "rework consumed {pct} percent of the remaining budget",
            "snagging and defect lists doubled in {weeks} weeks",
        ],
        "queries": [
            "defect counts are exploding and the client is rejecting deliverables",
            "quality is poor and we keep reworking the same items",
            "too many bugs are reaching the customer",
        ],
        "keywords": ["defect", "defects", "bugs", "bug", "quality", "rework", "snagging", "rejecting",
                     "failures", "faults", "inspection"],
    },
    "communication_breakdown": {
        "name": "Communication breakdown",
        "risk_category": "Organisational",
        "root_causes": ["Distributed teams without shared cadence", "Multiple conflicting status reports",
                        "Escalation routes undefined", "Information held in disconnected tools"],
        "symptoms": [
            "{deps} teams were working from different versions of the plan",
            "critical risks took {weeks} weeks to reach the steering group",
            "status reports contradicted each other in {changes} successive cycles",
        ],
        "queries": [
            "teams are working from different plans and nobody knows the real status",
            "bad news never reaches leadership until it is too late",
            "information is lost between our distributed teams",
        ],
        "keywords": ["communication", "status", "reporting", "information", "silos", "distributed",
                     "nobody knows", "escalation", "transparency", "visibility"],
    },
    "estimation_error": {
        "name": "Estimation error",
        "risk_category": "Schedule",
        "root_causes": ["Estimates anchored on the sales proposal", "No historical reference data used",
                        "Planning fallacy in bottom up estimates", "Complexity underestimated at approval"],
        "symptoms": [
            "actual effort ran {pct} percent above the approved estimate",
            "{changes} work packages exceeded their estimates by more than half",
            "the approved plan assumed productivity {pct} percent above industry norms",
        ],
        "queries": [
            "our original estimates were far too optimistic and the plan is not credible",
            "everything is taking much longer than we estimated",
            "the business case was approved on numbers that were unrealistic",
        ],
        "keywords": ["estimate", "estimates", "estimation", "estimated", "optimistic", "unrealistic",
                     "underestimated", "not credible", "longer than", "forecast"],
    },
    "change_resistance": {
        "name": "Change resistance",
        "risk_category": "Organisational",
        "root_causes": ["End users not involved in design", "Benefits never explained to frontline staff",
                        "Training delivered too late", "Union and workforce concerns unaddressed"],
        "symptoms": [
            "adoption of the new process reached only {pct} percent of target users",
            "{deps} business units continued to run legacy workarounds",
            "user complaints rose sharply in the {weeks} weeks after launch",
        ],
        "queries": [
            "users refuse to adopt the new system and keep using the old workarounds",
            "frontline staff are resisting the new way of working",
            "adoption after launch is far below target",
        ],
        "keywords": ["adoption", "adopt", "resistance", "resisting", "users refuse", "frontline",
                     "training", "culture", "workarounds", "change management", "take up"],
    },
}

INTERVENTIONS = {
    "change_control": {
        "name": "Formal change control board with impact triage",
        "mechanism": "every request was assessed for cost, time and benefit before approval",
        "keywords": ["change control", "change board", "impact assessment"],
    },
    "scope_rebaseline": {
        "name": "Scope rebaselining with sponsor sign off",
        "mechanism": "the scope was reprioritised and a new baseline was agreed with the sponsor",
        "keywords": ["rebaseline", "descope", "reprioritise"],
    },
    "critical_chain": {
        "name": "Critical chain buffer management",
        "mechanism": "task level padding was pooled into project and feeding buffers monitored weekly",
        "keywords": ["critical chain", "buffer"],
    },
    "fast_tracking": {
        "name": "Fast tracking of parallel work packages",
        "mechanism": "sequential activities were overlapped to recover time",
        "keywords": ["fast tracking", "parallel"],
    },
    "crashing_specialists": {
        "name": "Crashing with targeted specialist resources",
        "mechanism": "scarce specialists were added only to critical path activities",
        "keywords": ["crashing", "specialists"],
    },
    "supplier_recovery": {
        "name": "Supplier recovery plan with contractual remedies",
        "mechanism": "a joint recovery plan was tied to performance clauses and weekly reviews",
        "keywords": ["supplier recovery", "contractual remedies"],
    },
    "dual_sourcing": {
        "name": "Dual sourcing of critical supply",
        "mechanism": "a second qualified supplier was onboarded for the critical items",
        "keywords": ["dual sourcing", "second supplier"],
    },
    "sponsor_reset": {
        "name": "Executive sponsor reset workshop",
        "mechanism": "sponsors agreed a single set of outcomes, decision rights and tolerances",
        "keywords": ["sponsor reset", "executive workshop"],
    },
    "stakeholder_plan": {
        "name": "Stakeholder mapping and tailored engagement plan",
        "mechanism": "each stakeholder group received a tailored engagement route and owner",
        "keywords": ["stakeholder mapping", "engagement plan"],
    },
    "requirements_sprint": {
        "name": "Requirements discovery sprint with prototypes",
        "mechanism": "prototypes were tested with real users to settle acceptance criteria",
        "keywords": ["discovery sprint", "prototype", "prototypes"],
    },
    "tech_debt_sprints": {
        "name": "Dedicated technical debt reduction sprints",
        "mechanism": "capacity was ringfenced to refactor hotspots and add automated tests",
        "keywords": ["technical debt sprint", "refactoring sprint"],
    },
    "integration_harness": {
        "name": "Integration test harness and contract testing",
        "mechanism": "interfaces were locked with contract tests running in a shared environment",
        "keywords": ["contract testing", "test harness", "integration environment"],
    },
    "evm_control": {
        "name": "Earned value based cost control",
        "mechanism": "earned value metrics exposed variance early and triggered corrective action",
        "keywords": ["earned value", "evm", "cost control"],
    },
    "regulator_engagement": {
        "name": "Early regulator engagement and presubmission meetings",
        "mechanism": "the regulator reviewed draft submissions before formal filing",
        "keywords": ["regulator engagement", "presubmission"],
    },
    "knowledge_transfer": {
        "name": "Knowledge transfer and pairing programme",
        "mechanism": "critical knowledge was documented and shared through structured pairing",
        "keywords": ["knowledge transfer", "pairing", "succession"],
    },
    "quality_gates": {
        "name": "Quality gates with root cause analysis",
        "mechanism": "stage gates blocked progression until defect root causes were closed",
        "keywords": ["quality gate", "quality gates", "root cause analysis"],
    },
    "daily_standups": {
        "name": "Cross functional daily standups and a single source of truth dashboard",
        "mechanism": "all teams worked from one live dashboard reviewed in a short daily cadence",
        "keywords": ["standup", "standups", "dashboard", "single source of truth"],
    },
    "reference_class": {
        "name": "Reference class forecasting reestimation",
        "mechanism": "estimates were recalibrated against outcomes from comparable past projects",
        "keywords": ["reference class", "reestimate", "reestimation"],
    },
    "adoption_champions": {
        "name": "Change management and adoption champions network",
        "mechanism": "trained champions in every team supported users through the transition",
        "keywords": ["champions", "change management", "adoption plan"],
    },
    "resource_levelling": {
        "name": "Resource levelling with portfolio prioritisation",
        "mechanism": "portfolio leadership ranked initiatives and protected capacity for this one",
        "keywords": ["resource levelling", "portfolio prioritisation"],
    },
    "general_headcount": {
        "name": "Adding general headcount to the team",
        "mechanism": "additional generalist staff were onboarded across the team",
        "keywords": ["add headcount", "more people", "hire more"],
    },
    "overtime_push": {
        "name": "Sustained overtime push",
        "mechanism": "the team worked extended hours for several weeks",
        "keywords": ["overtime", "extended hours", "weekend working"],
    },
}

EFFICACY_TIERS = {"strong": 0.85, "moderate": 0.55, "neutral": 0.25, "harmful": 0.08}

EFFICACY_MAP = {
    "scope_creep": {"strong": ["change_control", "scope_rebaseline"],
                    "moderate": ["requirements_sprint", "sponsor_reset"], "harmful": ["overtime_push"]},
    "resource_contention": {"strong": ["resource_levelling", "crashing_specialists"],
                            "moderate": ["sponsor_reset", "fast_tracking"], "harmful": ["overtime_push"]},
    "critical_path_slippage": {"strong": ["critical_chain", "fast_tracking"],
                               "moderate": ["crashing_specialists", "reference_class"],
                               "harmful": ["general_headcount"]},
    "supplier_underperformance": {"strong": ["supplier_recovery", "dual_sourcing"],
                                  "moderate": ["daily_standups", "evm_control"], "harmful": ["overtime_push"]},
    "stakeholder_misalignment": {"strong": ["sponsor_reset", "stakeholder_plan"],
                                 "moderate": ["daily_standups", "scope_rebaseline"], "harmful": []},
    "requirements_ambiguity": {"strong": ["requirements_sprint", "stakeholder_plan"],
                               "moderate": ["change_control", "scope_rebaseline"], "harmful": ["fast_tracking"]},
    "technical_debt": {"strong": ["tech_debt_sprints", "quality_gates"],
                       "moderate": ["integration_harness"], "harmful": ["overtime_push", "fast_tracking"]},
    "integration_failure": {"strong": ["integration_harness", "quality_gates"],
                            "moderate": ["daily_standups", "crashing_specialists"], "harmful": ["fast_tracking"]},
    "budget_overrun": {"strong": ["evm_control", "scope_rebaseline"],
                       "moderate": ["reference_class", "supplier_recovery"],
                       "harmful": ["crashing_specialists", "general_headcount"]},
    "regulatory_delay": {"strong": ["regulator_engagement"],
                         "moderate": ["critical_chain", "stakeholder_plan"], "harmful": ["fast_tracking"]},
    "key_person_dependency": {"strong": ["knowledge_transfer"],
                              "moderate": ["resource_levelling", "crashing_specialists"],
                              "harmful": ["overtime_push"]},
    "quality_defects": {"strong": ["quality_gates", "tech_debt_sprints"],
                        "moderate": ["integration_harness"], "harmful": ["overtime_push", "fast_tracking"]},
    "communication_breakdown": {"strong": ["daily_standups", "stakeholder_plan"],
                                "moderate": ["sponsor_reset"], "harmful": []},
    "estimation_error": {"strong": ["reference_class", "scope_rebaseline"],
                         "moderate": ["evm_control", "critical_chain"], "harmful": ["overtime_push"]},
    "change_resistance": {"strong": ["adoption_champions", "stakeholder_plan"],
                          "moderate": ["sponsor_reset"], "harmful": []},
}

ISSUE_KEYS = list(ISSUES.keys())
INTERVENTION_KEYS = list(INTERVENTIONS.keys())
ISSUE_NAMES = [ISSUES[k]["name"] for k in ISSUE_KEYS]
INTERVENTION_NAMES = [INTERVENTIONS[k]["name"] for k in INTERVENTION_KEYS]
ISSUE_BY_NAME = {ISSUES[k]["name"]: k for k in ISSUE_KEYS}
INTERVENTION_BY_NAME = {INTERVENTIONS[k]["name"]: k for k in INTERVENTION_KEYS}


def efficacy_tier(issue_key, intervention_key):
    """Return the ground truth tier of an intervention for an issue."""
    tiers = EFFICACY_MAP[issue_key]
    for tier in ("strong", "moderate", "harmful"):
        if intervention_key in tiers[tier]:
            return tier
    return "neutral"


def efficacy_value(issue_key, intervention_key):
    return EFFICACY_TIERS[efficacy_tier(issue_key, intervention_key)]


def efficacy_matrix():
    """Dense matrix of efficacy values, issues by interventions."""
    return [[efficacy_value(i, j) for j in INTERVENTION_KEYS] for i in ISSUE_KEYS]
