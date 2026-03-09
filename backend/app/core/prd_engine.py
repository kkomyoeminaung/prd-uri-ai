"""
PRD Causal Engine — URI Theory Implementation
Myo Min Aung, February 2026

Implements:
  • ω_i(L) scale-dependent causal weighting   [eq. 3 in paper]
  • α ≈ 1.274 relational correction constant  [eq. 4-5 in paper]
  • Ψ_out = Σ ω_i Ĝ_i Ψ_in causal transform  [eq. 6 in paper]
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from .su5_generators import SU5Generators, PACCAYA_NAMES

logger = logging.getLogger(__name__)

# ── Physical constants ────────────────────────────────────────────────────────
PLANCK_LENGTH   = 1.616e-35   # meters
ALPHA_RELATIONAL = 1.274      # relational deviation constant (Kerr horizon integral)

# Relational eigenvalues λ_i for each Paccaya (derived from PRD symmetry)
LAMBDA_VALUES = {
    "hetu":          1.0,   # root — strongest causal driver
    "nissaya":       0.8,   # background support
    "indriya":       0.7,   # governing constraints
    "avigata":       0.9,   # stability invariance
    "anantara_12":   0.5,
    "anantara_13":   0.5,
    "anantara_14":   0.5,
    "anantara_15":   0.5,
    "anantara_23":   0.4,
    "anantara_24":   0.4,
    "anantara_25":   0.4,
    "anantara_34":   0.3,
    "anantara_35":   0.3,
    "anantara_45":   0.3,
    "sahajata_12":   0.6,   # symmetric interaction
    "sahajata_13":   0.6,
    "annamanna_12":  0.55,  # anti-symmetric interaction
    "annamanna_13":  0.55,
    # vigata (step-down) — filled from step_up mirror
    "vigata_21":     0.5,
    "vigata_31":     0.5,
    "vigata_41":     0.5,
    "vigata_51":     0.5,
    "vigata_32":     0.4,
    "vigata_42":     0.4,
}


class PRDCausalEngine:
    """
    Core causal engine implementing full URI-AGI transform:
        Ψ_out = Σ_{i=1}^{24} ω_i(L) · Ĝ_i · Ψ_in
    """

    def __init__(self):
        self.generators  = SU5Generators.get_all()          # 24 exact matrices
        self.lambda_vals = list(LAMBDA_VALUES.values())[:24]
        self.alpha       = ALPHA_RELATIONAL
        self._verify()

    def _verify(self):
        ok_h = SU5Generators.verify_hermitian(self.generators)
        ok_t = SU5Generators.verify_traceless(self.generators)
        logger.info(f"SU(5) verify — Hermitian: {ok_h}, Traceless: {ok_t}")

    # ── ω_i(L) weighting  [paper eq. 3] ──────────────────────────────────────
    def omega(self, scale_L: float) -> np.ndarray:
        """
        ω_i(L) = (1/Z) · exp(−λ_i / (L / l_P))
        scale_L : characteristic length of context (meters or normalised 0..1)
        """
        lP = PLANCK_LENGTH
        ratio = max(scale_L, lP) / lP          # L / l_P  (≥ 1)
        raw   = np.array([
            np.exp(-lam / ratio) for lam in self.lambda_vals
        ])
        Z = raw.sum() + 1e-12
        return raw / Z                          # normalised weights

    # ── α correction  [paper eq. 4-5] ────────────────────────────────────────
    def hawking_correction(self, horizon_area: float) -> float:
        """
        T_PRD = T_H · (1 + α · l_P² / A)
        Returns the multiplicative correction factor.
        """
        lP2 = PLANCK_LENGTH ** 2
        return 1.0 + self.alpha * lP2 / max(horizon_area, lP2)

    # ── Text → 5-D causal vector ──────────────────────────────────────────────
    def text_to_vector(self, text: str) -> np.ndarray:
        """Embed text into ℂ⁵ via keyword-weighted Paccaya projection."""
        KEYWORD_MAP = {
            "hetu":      ["cause", "root", "reason", "why", "origin", "because"],
            "nissaya":   ["support", "help", "background", "base", "foundation"],
            "indriya":   ["govern", "control", "faculty", "rule", "law"],
            "avigata":   ["stable", "constant", "invariant", "always", "permanent"],
            "anantara":  ["next", "follow", "sequence", "after", "then", "step"],
            "vigata":    ["end", "cease", "stop", "past", "done"],
            "sahajata":  ["together", "joint", "mutual", "both", "shared"],
            "annamanna": ["oppose", "contrast", "different", "versus", "anti"],
        }
        vec = np.zeros(5, dtype=complex)
        words = text.lower().split()
        for paccaya, keywords in KEYWORD_MAP.items():
            score = sum(1 for w in words if any(k in w for k in keywords))
            if score > 0:
                idx = list(KEYWORD_MAP.keys()).index(paccaya) % 5
                vec[idx] += score
        norm = np.linalg.norm(vec)
        return vec / (norm + 1e-12)

    # ── Main causal transform  [paper eq. 6] ──────────────────────────────────
    def transform(self, psi_in: np.ndarray, scale_L: float = 1e-10) -> np.ndarray:
        """
        Ψ_out = Σ ω_i · Ĝ_i · Ψ_in
        """
        weights   = self.omega(scale_L)
        causal_M  = sum(w * G for w, G in zip(weights, self.generators))
        psi_out   = causal_M @ psi_in
        norm      = np.linalg.norm(psi_out)
        return psi_out / (norm + 1e-12)

    # ── Named weight dict (for API responses) ─────────────────────────────────
    def compute_causal_weights(self, text: str, scale_L: float = 1e-10) -> Dict[str, float]:
        weights = self.omega(scale_L)
        names   = PACCAYA_NAMES + [f"gen_{i}" for i in range(24 - len(PACCAYA_NAMES))]
        return {name: float(np.real(w)) for name, w in zip(names[:24], weights)}

    # ── Dominant Paccaya analysis ──────────────────────────────────────────────
    def dominant_paccaya(self, text: str, scale_L: float = 1e-10, top_k: int = 3) -> List[Tuple[str, float]]:
        weights = self.compute_causal_weights(text, scale_L)
        sorted_w = sorted(weights.items(), key=lambda x: -x[1])
        return sorted_w[:top_k]
