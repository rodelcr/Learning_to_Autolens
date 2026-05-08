#!/usr/bin/env python3
"""h0_chain_overlay.py — overlay the three H0 posteriors landed 2026-05-08.

Loads the H0 column + Nautilus weights from each samples.csv and plots a
weighted kernel-density estimate. The three fits share a single mock
(`mocks_with_host/`) so the comparison is apples-to-apples — only the
likelihood changes:

    1. positions-only       (results/phase_4_positions_only_v2/)
    2. image-plane only     (results/phase_3_h0_free_tight/)
    3. joint image + Δt + positions (results/joint_h0_free/)

Output: figures/h0_chain_overlay.png. The plot is referenced by the
README's H0-chain results table and is the empirical companion to
Module 12 §3 (D_Δt cosmography) and §5 (TDCOSMO chain).

Usage:
    python Examples/quad_time_delay/h0_chain_overlay.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
QTD = REPO / "Examples" / "quad_time_delay"
RESULTS = QTD / "results"

H0_TRUTH = 70.0

FITS = [
    {
        "label": "Positions only (Track B v2)",
        "samples": RESULTS / "phase_4_positions_only_v2" / "samples.csv",
        "h0_col": "cosmology.H0",
        "color": "tab:gray",
    },
    {
        "label": "Image-plane only (Phase 3)",
        "samples": RESULTS / "phase_3_h0_free_tight" / "samples.csv",
        "h0_col": "cosmology.H0",
        "color": "tab:orange",
    },
    {
        "label": "Joint image + Δt + positions (Track D)",
        "samples": RESULTS / "joint_h0_free" / "samples.csv",
        "h0_col": "1.cosmology.H0",
        "color": "tab:blue",
    },
]


def weighted_quantile(values: np.ndarray, weights: np.ndarray, q: float) -> float:
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cdf = np.cumsum(w) / w.sum()
    return float(np.interp(q, cdf, v))


def weighted_kde(values: np.ndarray, weights: np.ndarray, grid: np.ndarray, bw: float) -> np.ndarray:
    diff = (grid[:, None] - values[None, :]) / bw
    kernel = np.exp(-0.5 * diff**2) / (bw * np.sqrt(2 * np.pi))
    return (kernel * weights[None, :]).sum(axis=1) / weights.sum()


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    grid = np.linspace(40, 130, 800)

    summary = []
    for fit in FITS:
        df = pd.read_csv(fit["samples"], skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        col = fit["h0_col"].strip()
        h0 = df[col].to_numpy()
        w = df["weight"].to_numpy() if "weight" in df.columns else np.ones_like(h0)
        w = w / w.sum()

        med = weighted_quantile(h0, w, 0.5)
        lo = weighted_quantile(h0, w, 0.16)
        hi = weighted_quantile(h0, w, 0.84)
        sigma = 0.5 * (hi - lo)

        bw = 1.06 * np.sqrt(((h0 - (h0 * w).sum()) ** 2 * w).sum()) * len(h0) ** -0.2
        bw = max(bw, 0.3)
        density = weighted_kde(h0, w, grid, bw)
        density /= density.max()

        ax.plot(grid, density, label=f"{fit['label']}", color=fit["color"], lw=2.2)
        ax.axvline(med, color=fit["color"], linestyle="--", alpha=0.4, lw=1)
        summary.append((fit["label"], med, sigma, lo, hi))

    ax.axvline(H0_TRUTH, color="black", lw=1.3, label=f"Truth H0 = {H0_TRUTH}")
    ax.set_xlabel("H0 [km/s/Mpc]")
    ax.set_ylabel("posterior density (peak-normalised)")
    ax.set_title("H0 chain — same mock, three likelihoods\n"
                 "Joint TDCOSMO methodology narrows σ(H0) ~10× over positions-only")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_xlim(40, 130)
    ax.set_ylim(bottom=0)
    fig.tight_layout()

    fig_dir = QTD / "figures"
    fig_dir.mkdir(exist_ok=True)
    out = fig_dir / "h0_chain_overlay.png"
    fig.savefig(out, dpi=140)
    print(f"[h0-chain] wrote {out.relative_to(REPO)}")

    print("\nSummary (median, ±1σ width, 16th, 84th):")
    for label, med, sigma, lo, hi in summary:
        bias = med - H0_TRUTH
        print(f"  {label}")
        print(f"      H0 = {med:6.2f}  σ ≈ {sigma:5.2f}  "
              f"[{lo:.2f}, {hi:.2f}]  bias = {bias:+.2f}")


if __name__ == "__main__":
    main()
