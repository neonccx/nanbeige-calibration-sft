"""Calibrated ideal notch response and observation-only fitting.

Response: Probst et al., RSI 86, 024706 (2015), Eq. (1),
https://arxiv.org/html/1410.3365v2, with a=1, alpha=tau=phi=0.
This is NOT the paper's full algebraic circle-fit/calibration algorithm.
Noise priors and quality thresholds below are project choices, not paper claims.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

SOURCE = "https://arxiv.org/html/1410.3365v2#S1.E1"
VERSION = "ideal-notch-pilot-0.1"


def notch(frequency_hz, resonance_hz: float, loaded_q: float, depth: float):
    f = np.asarray(frequency_hz, dtype=float)
    if (not np.isfinite(f).all() or np.any(f <= 0)
            or not np.isfinite([resonance_hz, loaded_q, depth]).all()
            or resonance_hz <= 0 or loaded_q <= 0 or not 0 < depth < 1):
        raise ValueError("Positive finite frequencies/Q and 0 < depth < 1 required")
    return 1 - depth / (1 + 2j * loaded_q * (f / resonance_hz - 1))


def measure(resonance_hz: float, loaded_q: float, depth: float,
            *, center_hz: float, span_hz: float, seed: int,
            noise_std: float = 0.002, points: int = 241) -> dict:
    if (not np.isfinite([center_hz, span_hz, noise_std]).all()
            or not 6e9 <= center_hz <= 7e9 or not 1e6 <= span_hz <= 100e6
            or noise_std < 0 or isinstance(points, bool) or not isinstance(points, int)
            or not 32 <= points <= 2001):
        raise ValueError("Invalid bounded pilot acquisition settings")
    f = np.linspace(center_hz - span_hz / 2, center_hz + span_hz / 2, points)
    z = notch(f, resonance_hz, loaded_q, depth)
    rng = np.random.default_rng(seed)
    z += noise_std * (rng.standard_normal(points) + 1j * rng.standard_normal(points))
    return {"schema_version": VERSION, "synthetic": True,
            "model": "calibrated_ideal_notch", "source": SOURCE,
            "frequency_hz": f.tolist(), "i": z.real.tolist(), "q": z.imag.tolist(),
            "acquisition": {"center_hz": center_hz, "span_hz": span_hz,
                            "points": points, "noise_std_per_quadrature": noise_std}}


def fit(observation: dict) -> dict:
    """Estimate from observations only; no simulator truth argument is accepted."""
    if observation.get("model") != "calibrated_ideal_notch":
        raise ValueError("This fitter requires calibrated ideal-notch observations")
    f = np.asarray(observation["frequency_hz"], dtype=float)
    i, q = (np.asarray(observation[key], dtype=float) for key in ("i", "q"))
    if (f.ndim != 1 or not 32 <= f.size <= 2001 or i.shape != f.shape or q.shape != f.shape
            or not np.isfinite(np.r_[f, i, q]).all() or np.any(f <= 0)
            or np.any(np.diff(f) <= 0)):
        raise ValueError("Finite, ordered, shape-aligned observations required")
    z = i + 1j * q
    center, scale = float((f[0] + f[-1]) / 2), float((f[-1] - f[0]) / 2)
    signal = float(np.sum(np.abs(z - z.mean()) ** 2))
    if signal < 1e-14 or np.max(np.abs(z - 1)) < 0.01:
        return {"reliable": False, "reason": "insufficient_signal", "fit": {},
                "analysis_version": VERSION}

    def residual(p):
        prediction = notch(f, center + p[0] * scale, np.exp(p[1]), p[2])
        error = prediction - z
        return np.r_[error.real, error.imag]

    candidates = []
    observed_center = float((f[np.argmin(abs(z))] - center) / scale)
    for initial_q in (1000, 4000, 12000):
        result = least_squares(residual, [np.clip(observed_center, -0.99, 0.99), np.log(initial_q), 0.5],
                               bounds=([-1, np.log(100), 0.01], [1, np.log(1e6), 0.99]),
                               max_nfev=1000)
        if result.success and np.isfinite(result.x).all():
            candidates.append(result)
    if not candidates:
        return {"reliable": False, "reason": "fit_failed", "fit": {},
                "analysis_version": VERSION}
    best = min(candidates, key=lambda result: float(np.sum(result.fun ** 2)))
    resonance = float(center + best.x[0] * scale)
    loaded_q, depth = float(np.exp(best.x[1])), float(best.x[2])
    rss = float(np.sum(best.fun ** 2))
    score = 1 - rss / signal
    width = resonance / loaded_q
    edge = abs(resonance - center) > 0.8 * scale
    covered = 2 * scale >= 4 * width
    resolved = float(np.max(np.diff(f))) <= width / 5
    bound_hit = bool(np.any(best.active_mask))
    checks = {"fit_r2_at_least_0_95": score >= 0.95, "not_at_scan_edge": not edge,
              "at_least_four_linewidths": covered, "at_least_five_points_per_linewidth": resolved,
              "not_at_fit_bound": not bound_hit}
    return {"reliable": bool(all(checks.values())),
            "reason": "quality_pass" if all(checks.values()) else "quality_rejected",
            "fit": {"resonance_hz": resonance, "loaded_q": loaded_q,
                    "depth": depth, "linewidth_hz": width},
            "diagnostics": {"fit_r2": float(score), "residual_rms": float(np.sqrt(rss / (2 * len(f)))),
                            "degrees_of_freedom": 2 * len(f) - 3, "checks": checks},
            "uncertainty": {"status": "not_implemented", "note": "Pilot is not production-ready"},
            "analysis_version": VERSION, "model_source": SOURCE}
