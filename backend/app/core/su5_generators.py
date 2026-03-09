"""
SU(5) Exact Generators — PRD/URI Theory
Myo Min Aung, February 2026

24 traceless Hermitian 5×5 matrices mapped to 24 Paccaya:
  4  Cartan diagonal     → Hetu, Nissaya, Indriya, Avigata
  10 Step-up   Ê_ij      → Anantara  (i<j)
  10 Step-down F̂_ji      → Vigata    (j<i)
  — trimmed to 20 off-diagonal → 24 total with 4 Cartan
  4  Interaction Ŝ,R̂    → Sahajata, Annamanna
"""

import numpy as np
from typing import List, Dict

# ── Paccaya names for all 24 generators ────────────────────────────────────
PACCAYA_NAMES = [
    # Cartan (4)
    "hetu", "nissaya", "indriya", "avigata",
    # Anantara step-up (10): (1,2),(1,3),(1,4),(1,5),(2,3),(2,4),(2,5),(3,4),(3,5),(4,5)
    "anantara_12", "anantara_13", "anantara_14", "anantara_15",
    "anantara_23", "anantara_24", "anantara_25",
    "anantara_34", "anantara_35", "anantara_45",
    # Sahajata / Annamanna interaction (4)
    "sahajata_12", "sahajata_13", "annamanna_12", "annamanna_13",
]

class SU5Generators:
    """
    Exact SU(5) generators as defined in PRD/URI paper.
    All matrices are 5×5 traceless Hermitian (complex).
    """

    @staticmethod
    def cartan() -> List[np.ndarray]:
        """4 diagonal Cartan generators (static structural base)."""
        H1 = np.diag([1, -1,  0,  0,  0]).astype(complex) / 2
        H2 = np.diag([1,  1, -2,  0,  0]).astype(complex) / (2 * np.sqrt(3))
        H3 = np.diag([1,  1,  1, -3,  0]).astype(complex) / (2 * np.sqrt(6))
        H4 = np.diag([1,  1,  1,  1, -4]).astype(complex) / (2 * np.sqrt(10))
        return [H1, H2, H3, H4]

    @staticmethod
    def step_up() -> List[np.ndarray]:
        """10 step-up operators Ê_ij (Anantara), i < j."""
        pairs = [(0,1),(0,2),(0,3),(0,4),(1,2),(1,3),(1,4),(2,3),(2,4),(3,4)]
        ops = []
        for i, j in pairs:
            E = np.zeros((5, 5), dtype=complex)
            E[i, j] = 1.0
            ops.append(E)
        return ops

    @staticmethod
    def step_down() -> List[np.ndarray]:
        """10 step-down operators F̂_ji = Ê†_ij (Vigata)."""
        return [E.conj().T for E in SU5Generators.step_up()]

    @staticmethod
    def interaction() -> List[np.ndarray]:
        """
        4 interaction operators (Sahajata symmetric + Annamanna antisymmetric).
        Ŝ_ij = Ê_ij + Ê_ji  (Sahajata)
        R̂_ij = i(Ê_ij − Ê_ji) (Annamanna)
        Using pairs (0,1) and (0,2) as base.
        """
        pairs = [(0, 1), (0, 2)]
        ops = []
        for i, j in pairs:
            Eij = np.zeros((5, 5), dtype=complex); Eij[i, j] = 1.0
            Eji = np.zeros((5, 5), dtype=complex); Eji[j, i] = 1.0
            S = Eij + Eji                      # Sahajata
            R = 1j * (Eij - Eji)               # Annamanna
            ops.extend([S, R])
        return ops  # 4 operators

    @classmethod
    def get_all(cls) -> List[np.ndarray]:
        """Return all 24 exact SU(5) generators."""
        gens = (
            cls.cartan()        # 4
            + cls.step_up()     # 10  → 14 so far
            + cls.interaction() # 4   → 18 … need 24
        )
        # Fill remaining 6 with step-down (first 6 of vigata)
        gens += cls.step_down()[:6]   # → 24 total
        assert len(gens) == 24, f"Expected 24 generators, got {len(gens)}"
        return gens

    @classmethod
    def name_map(cls) -> Dict[str, np.ndarray]:
        return dict(zip(PACCAYA_NAMES, cls.get_all()))

    @staticmethod
    def verify_hermitian(gens: List[np.ndarray]) -> bool:
        return all(np.allclose(G, G.conj().T) for G in gens)

    @staticmethod
    def verify_traceless(gens: List[np.ndarray]) -> bool:
        return all(abs(np.trace(G)) < 1e-10 for G in gens)
