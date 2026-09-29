"""REST API.

Start with: python gantry.py serve
Interactive documentation is served at /docs once running.

Set GANTRY_API_KEY to require an Authorization header of the form
"Bearer <key>" on every endpoint except health.
"""

import os
import time
from contextlib import asynccontextmanager
from typing import Dict, List, Optional, Union

from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from pydantic import BaseModel, Field

from . import __version__
from .engine import GantryEngine
from .utils import sub

STARTED = time.time()
FilterValue = Union[str, List[str]]


@asynccontextmanager
async def lifespan(app):
    GantryEngine.shared()
    yield


app = FastAPI(
    title="Gantry AI",
    version=__version__,
    description="Evidence grounded answers to complex project delivery problems, drawn from a base of "
                "historical project cases with verified citations.",
    lifespan=lifespan,
)
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.add_middleware(CORSMiddleware, allow_origins=os.environ.get("GANTRY_CORS", "*").split(","),
                   allow_methods=["GET", "POST"], allow_headers=["*"])


def require_key(authorization: Optional[str] = Header(default=None)):
    expected = os.environ.get("GANTRY_API_KEY")
    if not expected:
        return
    if authorization != "Bearer " + expected:
        raise HTTPException(status_code=401, detail="Missing or invalid bearer token.")


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=4000,
                          examples=["Our supplier keeps missing delivery dates on a fixed price build. What should we do?"])
    filters: Optional[Dict[str, FilterValue]] = Field(
        default=None, description="Hard filters on case metadata, for example {\"industry\": \"Energy\"}.")
    k: int = Field(default=8, ge=1, le=25, description="Number of precedent cases in the context.")
    provider: Optional[str] = Field(default=None, description="extractive, anthropic or ollama.")
    strict: bool = Field(default=True, description="Replace model answers that fail verification.")
    include_evidence_pack: bool = False


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, max_length=2000)
    filters: Optional[Dict[str, FilterValue]] = None
    k: int = Field(default=10, ge=1, le=100)
    mode: str = Field(default="hybrid", pattern="^(hybrid|fusion|bm25|dense)$")


class RecommendRequest(BaseModel):
    issue: str = Field(..., examples=["Scope creep"])
    industry: Optional[str] = None
    methodology: Optional[str] = None
    contract: Optional[str] = None


class AssessRequest(BaseModel):
    industry: str = "Software"
    delivery_methodology: str = "Hybrid"
    contract_type: str = "Fixed Price"
    organisation_size: str = "Enterprise"
    requirements_volatility: str = "Moderate"
    regulatory_exposure: str = "Medium"
    region: str = "UK and Ireland"
    technical_complexity: float = Field(default=6, ge=1, le=10)
    sponsor_engagement: float = Field(default=3, ge=1, le=5)
    pm_experience_years: float = Field(default=8, ge=0, le=50)
    team_size: float = Field(default=25, ge=1)
    vendor_count: float = Field(default=4, ge=0)
    dependency_count: float = Field(default=12, ge=0)
    stakeholder_count: float = Field(default=12, ge=0)
    team_turnover_pct: float = Field(default=10, ge=0, le=100)
    baseline_budget_gbp: float = Field(default=2_500_000, gt=0)
    planned_duration_days: float = Field(default=300, gt=0)


@app.get("/")
def root():
    return {"name": "Gantry AI", "version": __version__, "docs": "/docs"}


@app.get("/health")
def health():
    engine = GantryEngine.shared()
    return {"status": "ok", "cases": engine.index.size, "uptime_seconds": round(sub(time.time(), STARTED), 1)}


@app.get("/stats", dependencies=[Depends(require_key)])
def stats():
    return GantryEngine.shared().stats()


@app.get("/options", dependencies=[Depends(require_key)])
def options():
    return GantryEngine.shared().options()


@app.post("/ask", dependencies=[Depends(require_key)])
def ask(req: AskRequest):
    engine = GantryEngine.shared()
    result = engine.ask(req.question, filters=req.filters, k=req.k, provider=req.provider, strict=req.strict)
    result = dict(result)
    if req.include_evidence_pack:
        from .answer import evidence_pack
        result["evidence_pack"] = evidence_pack(req.question, result)
    return result


@app.post("/search", dependencies=[Depends(require_key)])
def search(req: SearchRequest):
    return GantryEngine.shared().search(req.query, filters=req.filters, k=req.k, mode=req.mode)


@app.post("/recommend", dependencies=[Depends(require_key)])
def recommend(req: RecommendRequest):
    try:
        return GantryEngine.shared().recommend(req.issue, req.industry, req.methodology, req.contract)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=str(exc).strip("'"))


@app.post("/assess", dependencies=[Depends(require_key)])
def assess(req: AssessRequest):
    return GantryEngine.shared().assess(req.model_dump())


@app.get("/cases/{project_id}", dependencies=[Depends(require_key)])
def case(project_id: str):
    found = GantryEngine.shared().case(project_id.upper())
    if not found:
        raise HTTPException(status_code=404, detail="No case with id " + project_id)
    return found
