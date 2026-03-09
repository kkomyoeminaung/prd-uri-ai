"""
Causal Analysis API — expose PRD/URI engine directly
"""

from fastapi import APIRouter
from pydantic import BaseModel
from typing import Optional
import numpy as np

from app.core.prd_engine import PRDCausalEngine, ALPHA_RELATIONAL, PLANCK_LENGTH
from app.core.counselor import CausalCounselor
from app.core.su5_generators import SU5Generators

router = APIRouter()
_engine    = PRDCausalEngine()
_counselor = CausalCounselor(_engine)


class AnalyzeRequest(BaseModel):
    text: str
    scale_L: float = 1e-10


@router.post("/analyze")
async def analyze(req: AnalyzeRequest):
    ctx = _counselor.analyze(req.text, req.scale_L)
    return {
        "dominant_paccaya":  ctx["dominant_paccaya"],
        "upanissaya_score":  ctx["upanissaya_score"],
        "asevana_score":     ctx["asevana_score"],
        "alpha_correction":  ctx["alpha_correction"],
        "psi_out_norm":      ctx["psi_out_norm"],
        "causal_weights":    ctx["causal_weights"],
    }


@router.get("/generators")
async def get_generators():
    """Return metadata about all 24 SU(5) generators."""
    from app.core.su5_generators import PACCAYA_NAMES
    gens = SU5Generators.get_all()
    return {
        "count": len(gens),
        "hermitian": SU5Generators.verify_hermitian(gens),
        "traceless":  SU5Generators.verify_traceless(gens),
        "paccaya_names": PACCAYA_NAMES,
        "alpha": ALPHA_RELATIONAL,
        "planck_length": PLANCK_LENGTH,
    }


@router.post("/omega")
async def compute_omega(scale_L: float = 1e-10):
    """Compute ω_i(L) weights at a given scale."""
    weights = _engine.omega(scale_L)
    from app.core.su5_generators import PACCAYA_NAMES
    return {
        "scale_L": scale_L,
        "weights": dict(zip(PACCAYA_NAMES, [float(w) for w in weights])),
    }


@router.post("/hawking")
async def hawking_correction(horizon_area: float = 1e-10):
    """Compute T_PRD / T_H correction factor for given horizon area."""
    correction = _engine.hawking_correction(horizon_area)
    return {
        "horizon_area":       horizon_area,
        "alpha":              ALPHA_RELATIONAL,
        "correction_factor":  correction,
        "formula": "T_PRD = T_H × (1 + α·l_P²/A)",
    }
