"""
URI Causal Counselor — Upanissaya → Asevana Shift
Myo Min Aung, February 2026

Based on URI paper Appendix A & Section IV-A:
  • Upanissaya = past habitual momentum (anusaya patterns)
  • Nissaya    = support / therapeutic scaffold
  • Asevana    = new intentional habitual momentum
  • Hetu       = root cause identification

The counselor diagnoses dominant Paccaya patterns in the user's
input, then applies a weight-shift to strengthen Nissaya (support)
and Asevana (new momentum) while reducing Upanissaya grip.
"""

import numpy as np
from typing import Dict, List, Tuple
import logging

from .prd_engine import PRDCausalEngine, PACCAYA_NAMES
from .su5_generators import SU5Generators

logger = logging.getLogger(__name__)

# ── Paccaya pattern → psychological meaning ───────────────────────────────────
PACCAYA_MEANING = {
    "hetu":          "Root cause / core driver of your experience",
    "nissaya":       "Support structures available to you",
    "indriya":       "Governing faculties (attention, intention, energy)",
    "avigata":       "Stable background conditions",
    "anantara":      "Sequential momentum — what naturally follows",
    "anantara_12":   "Primary sequential flow",
    "anantara_13":   "Secondary progression",
    "anantara_14":   "Tertiary unfolding",
    "anantara_15":   "Distal causal chain",
    "anantara_23":   "Mid-chain causation",
    "anantara_24":   "Extended causal reach",
    "anantara_25":   "Far causal extension",
    "anantara_34":   "Deep transition",
    "anantara_35":   "Subtle flow",
    "anantara_45":   "Terminal transition",
    "sahajata_12":   "Co-arising — things that reinforce each other",
    "sahajata_13":   "Mutual arising pattern",
    "annamanna_12":  "Oppositional tension — conflicting forces",
    "annamanna_13":  "Polarity / contrast dynamic",
    "vigata_21":     "Past conditions releasing",
    "vigata_31":     "Old patterns dissolving",
    "vigata_41":     "Deep past releasing",
    "vigata_51":     "Root past momentum ending",
    "vigata_32":     "Secondary past dissolving",
    "vigata_42":     "Extended past releasing",
}

# Upanissaya keywords (past habit, stuck patterns)
UPANISSAYA_KEYWORDS = [
    "trapped", "stuck", "always", "never", "habit", "past", "mistake",
    "regret", "can't", "cannot", "failed", "failure", "repeat", "again",
    "same", "pattern", "hopeless", "tired", "exhausted", "worthless",
    "blame", "guilt", "shame", "worry", "fear", "anxious", "depressed",
    "pain", "suffer", "lost", "broken", "wrong", "bad"
]

# Asevana keywords (new intention, growth)
ASEVANA_KEYWORDS = [
    "change", "grow", "new", "start", "begin", "try", "hope", "future",
    "better", "improve", "learn", "practice", "intention", "goal", "step",
    "forward", "heal", "release", "free", "possible", "can", "will", "want"
]


class CausalCounselor:
    """
    URI Causal Counseling System.
    Applies Upanissaya → Asevana weight shift via Nissaya support scaffold.
    """

    def __init__(self, prd_engine: PRDCausalEngine):
        self.engine = prd_engine
        self.generators = SU5Generators.get_all()
        self.session_history: List[Dict] = []

    # ── Detect Upanissaya load ────────────────────────────────────────────────
    def _upanissaya_score(self, text: str) -> float:
        """0.0–1.0: how much past-habit momentum is present."""
        words = text.lower().split()
        hits  = sum(1 for w in words if any(k in w for k in UPANISSAYA_KEYWORDS))
        return min(hits / max(len(words), 1) * 5, 1.0)

    def _asevana_score(self, text: str) -> float:
        """0.0–1.0: how much new-intention momentum is present."""
        words = text.lower().split()
        hits  = sum(1 for w in words if any(k in w for k in ASEVANA_KEYWORDS))
        return min(hits / max(len(words), 1) * 5, 1.0)

    # ── Weight shift: strengthen Nissaya + Asevana ────────────────────────────
    def _shift_weights(self, base_weights: Dict[str, float],
                       upanissaya_load: float) -> Dict[str, float]:
        """
        Apply relational weight-shift:
          • Boost  nissaya   by upanissaya_load × α
          • Boost  indriya   (governing intention)
          • Reduce vigata    (past releasing) slightly
          • Normalise
        """
        w = dict(base_weights)
        boost = upanissaya_load * self.engine.alpha * 0.1

        for k in w:
            if "nissaya" in k:
                w[k] = w.get(k, 0) + boost
            if "indriya" in k:
                w[k] = w.get(k, 0) + boost * 0.5
            if "vigata" in k:
                w[k] = max(w.get(k, 0) - boost * 0.3, 0.001)

        total = sum(w.values()) + 1e-12
        return {k: v / total for k, v in w.items()}

    # ── Build Ollama prompt with causal context ───────────────────────────────
    def build_system_prompt(self, causal_ctx: Dict) -> str:
        dom = causal_ctx["dominant_paccaya"]
        up  = causal_ctx["upanissaya_score"]
        ase = causal_ctx["asevana_score"]
        corr = causal_ctx["alpha_correction"]

        dominant_str = "\n".join(
            f"  • {name} ({score:.3f}): {PACCAYA_MEANING.get(name, name)}"
            for name, score in dom
        )

        return f"""You are URI-AGI, a causal intelligence system based on Pattana-Relational Dynamics (PRD) and SU(5) algebra.

You do NOT answer from statistical pattern-matching alone. You reason from CAUSAL STRUCTURE.

## Current Causal Analysis of User's Message
Dominant Paccaya (causal conditions active):
{dominant_str}

Upanissaya load (past habitual momentum): {up:.2f}/1.0
Asevana potential (new intentional momentum): {ase:.2f}/1.0
α-relational correction factor: {corr:.6f}

## Your Counseling Mode
{"⚠️  HIGH Upanissaya detected — apply Nissaya (support) scaffold. Gently help shift from stuck patterns toward new intention (Asevana)." if up > 0.3 else "ℹ️  Balanced causal state — respond with clear causal analysis."}

## Instructions
- Reference the dominant Paccaya naturally in your response
- If Upanissaya is high: acknowledge the pattern compassionately, then offer a Nissaya-supported path
- Be concise, warm, and grounded in causal logic
- End with one practical Asevana step the person can take now
"""

    # ── Main counsel method ───────────────────────────────────────────────────
    def analyze(self, user_input: str, scale_L: float = 1e-10) -> Dict:
        """
        Full causal analysis of user input.
        Returns structured context for Ollama prompt injection.
        """
        # Causal transform
        psi_in  = self.engine.text_to_vector(user_input)
        psi_out = self.engine.transform(psi_in, scale_L)

        # Weights + shift
        base_weights  = self.engine.compute_causal_weights(user_input, scale_L)
        up_score      = self._upanissaya_score(user_input)
        ase_score     = self._asevana_score(user_input)
        shifted       = self._shift_weights(base_weights, up_score)

        # Dominant Paccaya
        dominant = sorted(shifted.items(), key=lambda x: -x[1])[:3]

        # α correction (horizon area ≈ normalised text complexity)
        area    = len(user_input.split()) * 1e-10
        corr    = self.engine.hawking_correction(area)

        ctx = {
            "dominant_paccaya":  dominant,
            "causal_weights":    shifted,
            "upanissaya_score":  up_score,
            "asevana_score":     ase_score,
            "alpha_correction":  corr,
            "psi_out_norm":      float(np.linalg.norm(psi_out)),
            "system_prompt":     "",
        }
        ctx["system_prompt"] = self.build_system_prompt(ctx)

        # Store session
        self.session_history.append({
            "input": user_input,
            "upanissaya": up_score,
            "asevana": ase_score,
            "dominant": dominant,
        })

        return ctx

    def session_summary(self) -> Dict:
        """Trend analysis across conversation."""
        if not self.session_history:
            return {}
        up_trend  = [h["upanissaya"] for h in self.session_history]
        ase_trend = [h["asevana"]    for h in self.session_history]
        return {
            "turns":           len(self.session_history),
            "upanissaya_trend": up_trend,
            "asevana_trend":    ase_trend,
            "net_shift":        (ase_trend[-1] - up_trend[-1]) if len(up_trend) > 0 else 0,
            "improving":        ase_trend[-1] > up_trend[-1] if len(up_trend) > 0 else False,
        }
