"""plan-023 §3.4 — 4 large-N anchor layout codebook (Frenet, 단위 m, K > 14).

각 layout 의 invariant (smoke 검증):
  - dtype = np.float32
  - shape = (N, 3), N ∈ {20, 24, 30, 50}
  - max ‖a‖ ≤ 0.005m + 1e-7 (= 0.5cm radius bound, plan-022 carry)
  - B1/B2/B3 vertex-transitive: ‖a‖ std ≤ 1e-6 (모든 vertex norm 동일)
  - B4 quasi-uniform (Fibonacci spiral): 단위 sphere 위 → 모든 vertex norm = 0.005m exact
  - **unique vertex 검증**: np.unique(arr, axis=0).shape[0] == N
    (silent duplicate from cyclic-perm bug 차단)

Anchor index 일관성: 모든 layout 이 vertex-transitive 또는 quasi-uniform 으로 center 없음
→ index 0 = generator 의 첫 vertex (deterministic order).
plan-022 의 center-bearing layout (A1/A2/A3/A4/A7) 과 cross-plan index 매핑 불가.
"""
from __future__ import annotations

import itertools

import numpy as np


_PHI = (1.0 + np.sqrt(5.0)) / 2.0          # golden ratio ≈ 1.6180339
_INV_PHI = 1.0 / _PHI                       # = φ - 1 ≈ 0.6180339
_PHI2 = _PHI * _PHI                         # = φ + 1 ≈ 2.6180339


# ── B1: dodecahedron (20) — Platonic ──────────────────────────────────
#
# vertices = (±1, ±1, ±1) 8개 cube + (0, ±1/φ, ±φ) + 2 cyclic perm × 4 signs = 12.
# unscaled norm = √3 for all. scale = 0.005 / √3 ≈ 0.002887.

_cube8 = np.array(list(itertools.product([-1.0, 1.0], repeat=3)), dtype=np.float64)

_rect12 = []
for _sy, _sz in itertools.product([-1.0, 1.0], repeat=2):
    _rect12.append((0.0,             _sy * _INV_PHI, _sz * _PHI))      # (0, ±1/φ, ±φ)
    _rect12.append((_sy * _INV_PHI, _sz * _PHI,     0.0))              # (±1/φ, ±φ, 0)
    _rect12.append((_sz * _PHI,     0.0,             _sy * _INV_PHI))  # (±φ, 0, ±1/φ)
_rect12 = np.array(_rect12, dtype=np.float64)

_DODECA20 = np.vstack([_cube8, _rect12])                                # (20, 3) unscaled
ANCHORS_B1 = (_DODECA20 * (0.005 / np.sqrt(3.0))).astype(np.float32)    # (20, 3)


# ── B2: truncated octahedron (24) — Archimedean ───────────────────────
#
# vertices = all signed permutations of (0, ±1, ±2). norm = √5. scale = 0.005/√5.

_TRUNC_OCTA_verts: list[tuple[float, float, float]] = []
for _zero_pos in range(3):                                              # which position is 0
    _other = [i for i in range(3) if i != _zero_pos]
    for _swap in [(1.0, 2.0), (2.0, 1.0)]:                              # {±1, ±2} assignment
        for _s1, _s2 in itertools.product([-1.0, 1.0], repeat=2):
            _v = [0.0, 0.0, 0.0]
            _v[_other[0]] = _s1 * _swap[0]
            _v[_other[1]] = _s2 * _swap[1]
            _TRUNC_OCTA_verts.append((_v[0], _v[1], _v[2]))
_TRUNC_OCTA24 = np.array(_TRUNC_OCTA_verts, dtype=np.float64)           # (24, 3)
ANCHORS_B2 = (_TRUNC_OCTA24 * (0.005 / np.sqrt(5.0))).astype(np.float32)


# ── B3: icosidodecahedron (30) — Archimedean ──────────────────────────
#
# vertices = 6 axis ((±φ, 0, 0) cyclic perm) + 24 (±1/2, ±φ/2, ±φ²/2) cyclic perm.
# norm = φ for all (math: (1/4)(1 + φ² + φ⁴) = (1/4)·4φ² = φ²; axis trivially φ).
# scale = 0.005 / φ ≈ 0.003090.

_axis6 = np.array(
    [
        (+_PHI, 0.0, 0.0), (-_PHI, 0.0, 0.0),
        (0.0, +_PHI, 0.0), (0.0, -_PHI, 0.0),
        (0.0, 0.0, +_PHI), (0.0, 0.0, -_PHI),
    ],
    dtype=np.float64,
)

_cyclic24 = []
for _sx, _sy, _sz in itertools.product([-1.0, 1.0], repeat=3):
    _cyclic24.append((_sx * 0.5,        _sy * _PHI / 2.0,  _sz * _PHI2 / 2.0))
    _cyclic24.append((_sz * _PHI2 / 2.0, _sx * 0.5,        _sy * _PHI / 2.0))
    _cyclic24.append((_sy * _PHI / 2.0,  _sz * _PHI2 / 2.0, _sx * 0.5))
_cyclic24 = np.array(_cyclic24, dtype=np.float64)                       # (24, 3)

_ICOSIDODEC30 = np.vstack([_axis6, _cyclic24])                          # (30, 3)
ANCHORS_B3 = (_ICOSIDODEC30 * (0.005 / _PHI)).astype(np.float32)


# ── B4: Fibonacci spiral N=50 — quasi-uniform ─────────────────────────
#
# plan-022 §3.4 A7 의 fib_sphere generator 정확 carry, N=12 → N=50.

def _fib_sphere(N: int, r: float) -> np.ndarray:
    """N 점 Fibonacci spiral on sphere(radius=r). returns (N, 3)."""
    phi_g = (1.0 + np.sqrt(5.0)) / 2.0
    i = np.arange(N)
    theta = 2.0 * np.pi * i / phi_g                                     # azimuth
    z = 1.0 - 2.0 * (i + 0.5) / N                                       # latitude (uniform in z)
    rho = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    return np.stack([rho * np.cos(theta), rho * np.sin(theta), z], axis=1) * r


ANCHORS_B4 = _fib_sphere(50, 0.005).astype(np.float32)                  # (50, 3)


# ── LAYOUT_NAMES_B dict (export) ──────────────────────────────────────

LAYOUT_NAMES_B = {
    "B1_dodeca20":      ANCHORS_B1,
    "B2_trunc_octa24":  ANCHORS_B2,
    "B3_icosidodec30":  ANCHORS_B3,
    "B4_fib50":         ANCHORS_B4,
}


# ── smoke (module import 시 실행되는 invariant 검증, §4.5 carry) ──────

def _smoke() -> None:
    """plan-023 §3.4 / §4.5 T2 + T3 + unique invariants."""
    expected_K = {
        "B1_dodeca20": 20,
        "B2_trunc_octa24": 24,
        "B3_icosidodec30": 30,
        "B4_fib50": 50,
    }
    for name, arr in LAYOUT_NAMES_B.items():
        K = expected_K[name]
        # T2: dtype + shape + norm bound
        assert arr.dtype == np.float32, f"{name}: dtype {arr.dtype} != float32"
        assert arr.shape == (K, 3), f"{name}: shape {arr.shape} != ({K}, 3)"
        norms = np.linalg.norm(arr, axis=1)
        assert norms.max() <= 0.005 + 1e-7, \
            f"{name}: max ‖a‖ = {norms.max():.7f} > 0.005 + 1e-7"
        assert np.isfinite(arr).all(), f"{name}: non-finite entry"
        # T3: single-shell std ≤ 1e-6 (B1/B2/B3 vertex-transitive, B4 unit sphere)
        assert norms.std() <= 1e-6, \
            f"{name}: norm std {norms.std():.2e} > 1e-6 (single-shell violated)"
        # unique vertex 검증 (silent duplicate 차단)
        n_unique = np.unique(arr, axis=0).shape[0]
        assert n_unique == K, \
            f"{name}: unique vertex {n_unique} != {K} (cyclic-perm duplicate bug)"


if __name__ == "__main__":
    _smoke()
    for name, arr in LAYOUT_NAMES_B.items():
        norms = np.linalg.norm(arr, axis=1)
        print(
            f"{name:18s} K={arr.shape[0]:2d}  "
            f"‖a‖ min/max = {norms.min():.6f} / {norms.max():.6f}  "
            f"std = {norms.std():.2e}"
        )
