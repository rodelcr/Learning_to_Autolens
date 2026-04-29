"""Benchmark autolens log_likelihood for the multi-plane R5 model:
numpy (use_jax=False, 32-core multiprocessing) vs JAX on a single A100 GPU
vs JAX with pmap across 4 GPUs.

The single-plane probe (al_gpu_probe.py) found JAX-GPU 4× slower than numpy
on ~9k-pixel HST cutouts. The hypothesis here is that the multi-plane R5
model (recursive plane-to-plane ray-tracing through 2 deflectors + 2 source
components) has ~15× more compute per likelihood call, which might tip the
balance in favour of GPU.

Run on Cannon with --gres=gpu:1 (single-GPU baseline) or --gres=gpu:4
(multi-GPU pmap test).
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def build_R5_tracer(truths: dict):
    """Build the R5 tracer with truth-anchored parameters."""
    import autofit as af
    import autolens as al

    # Extract truths
    z_l1 = truths["redshifts"]["lens_primary"]
    z_l2 = truths["redshifts"]["lens_secondary"]
    z_s = truths["redshifts"]["source"]

    # Primary EPL (autolens uses ell_comps; truths use lenstronomy e1, e2)
    epls = [(i, kw) for i, (m, kw)
            in enumerate(zip(truths["lens_model_list"], truths["kwargs_lens"]))
            if m == "EPL"]
    primary, secondary = epls[0][1], epls[1][1]

    src_kwargs = truths.get("kwargs_source", [])
    light = truths.get("kwargs_lens_light", [{}])[0]

    # Build galaxies. We use truth values as-is — the convention drift
    # would change which axis is flipped, but for benchmark purposes the
    # likelihood evaluation cost is the same regardless of the optimum.
    bulge = al.lp.Sersic(
        centre=(light.get("center_x", 0.0), light.get("center_y", 0.0)),
        ell_comps=(light.get("e1", 0.0), light.get("e2", 0.0)),
        intensity=light.get("amp", 1.0),
        effective_radius=light.get("R_sersic", 1.0),
        sersic_index=light.get("n_sersic", 4.0),
    )

    # Use Isothermal (γ=2 fixed) for JAX compatibility — autolens's PowerLaw
    # JAX path has a bug in `omega()` (jax.lax.scan with functools.partial)
    # that crashes on first eval. Isothermal is the JAX-compatible variant
    # and gives us the multi-plane likelihood timing we want.
    use_powerlaw = os.environ.get("PROBE_USE_POWERLAW", "0") == "1"
    if use_powerlaw:
        mass_1 = al.mp.PowerLaw(
            centre=(primary.get("center_x", 0.0), primary.get("center_y", 0.0)),
            ell_comps=(primary.get("e1", 0.0), primary.get("e2", 0.0)),
            einstein_radius=primary.get("theta_E", 1.0),
            slope=primary.get("gamma", 2.0),
        )
    else:
        mass_1 = al.mp.Isothermal(
            centre=(primary.get("center_x", 0.0), primary.get("center_y", 0.0)),
            ell_comps=(primary.get("e1", 0.0), primary.get("e2", 0.0)),
            einstein_radius=primary.get("theta_E", 1.0),
        )

    # Convert lenstronomy SHEAR_GAMMA_PSI to Cartesian for autolens
    import math
    shear_kwargs = next((kw for m, kw in zip(truths["lens_model_list"],
                                              truths["kwargs_lens"])
                         if m == "SHEAR_GAMMA_PSI"), None)
    if shear_kwargs:
        gx = shear_kwargs["gamma_ext"]
        psi = shear_kwargs["psi_ext"]
        g1 = gx * math.cos(2 * psi)
        g2 = gx * math.sin(2 * psi)
    else:
        g1, g2 = 0.0, 0.0
    shear = al.mp.ExternalShear(gamma_1=g1, gamma_2=g2)

    lens_1 = al.Galaxy(redshift=z_l1, bulge=bulge, mass=mass_1, shear=shear)

    mass_2 = al.mp.Isothermal(
        centre=(secondary.get("center_x", 0.0), secondary.get("center_y", 0.0)),
        ell_comps=(secondary.get("e1", 0.0), secondary.get("e2", 0.0)),
        einstein_radius=secondary.get("theta_E", 0.1),
    )
    lens_2 = al.Galaxy(redshift=z_l2, mass=mass_2)

    src_a_kw = src_kwargs[0] if len(src_kwargs) > 0 else {}
    src_b_kw = src_kwargs[1] if len(src_kwargs) > 1 else {}

    src_a = al.lp.SersicCore(
        centre=(src_a_kw.get("center_x", 0.0), src_a_kw.get("center_y", 0.0)),
        ell_comps=(src_a_kw.get("e1", 0.0), src_a_kw.get("e2", 0.0)),
        intensity=src_a_kw.get("amp", 1.0),
        effective_radius=src_a_kw.get("R_sersic", 0.2),
        sersic_index=src_a_kw.get("n_sersic", 1.0),
    )
    src_b = al.lp.SersicCore(
        centre=(src_b_kw.get("center_x", 0.0), src_b_kw.get("center_y", 0.0)),
        ell_comps=(src_b_kw.get("e1", 0.0), src_b_kw.get("e2", 0.0)),
        intensity=src_b_kw.get("amp", 1.0),
        effective_radius=src_b_kw.get("R_sersic", 0.2),
        sersic_index=src_b_kw.get("n_sersic", 1.0),
    )
    source = al.Galaxy(redshift=z_s, bulge=src_a, disk=src_b)

    return al.Tracer(galaxies=[lens_1, lens_2, source])


def load_dataset(dataset_root: Path, mock_index: int):
    import autolens as al
    dataset = al.Imaging.from_fits(
        data_path=dataset_root / f"lenstronomy_mock_{mock_index}_image.fits",
        noise_map_path=dataset_root / f"lenstronomy_mock_{mock_index}_noise.fits",
        psf_path=dataset_root / "lenstronomy_mock_psf.fits",
        pixel_scales=0.05,
    )
    mask = al.Mask2D.circular(
        shape_native=dataset.shape_native,
        pixel_scales=dataset.pixel_scales,
        radius=2.7,
    )
    return dataset.apply_mask(mask=mask)


def time_likelihood_numpy(dataset, tracer, n_warm: int = 7):
    import autolens as al
    print("=== use_jax=False (numpy) ===", flush=True)
    t0 = time.time()
    ll = al.FitImaging(dataset=dataset, tracer=tracer).log_likelihood
    print(f"first call: ll={ll:.2f}, time={time.time()-t0:.3f}s", flush=True)
    warm = []
    for _ in range(n_warm):
        t0 = time.time()
        _ = al.FitImaging(dataset=dataset, tracer=tracer).log_likelihood
        warm.append(time.time() - t0)
    warm.sort()
    median = warm[len(warm) // 2]
    print(f"warm {n_warm}x: {['%.3f' % t for t in warm]}, median={median*1000:.1f}ms",
          flush=True)
    return median


def time_likelihood_jax(dataset, tracer, n_warm: int = 7):
    import autolens as al
    import jax
    import jax.numpy as jnp
    print(f"\n=== use_jax=True (jax on {jax.default_backend()}) ===", flush=True)
    print(f"  devices: {jax.devices()}", flush=True)
    try:
        t0 = time.time()
        fit_jax = al.FitImaging(dataset=dataset, tracer=tracer, xp=jnp)
        ll = float(fit_jax.log_likelihood)
        print(f"first call (cold+jit): ll={ll:.2f}, time={time.time()-t0:.3f}s",
              flush=True)
        warm = []
        for _ in range(n_warm):
            t0 = time.time()
            _ = float(al.FitImaging(dataset=dataset, tracer=tracer,
                                    xp=jnp).log_likelihood)
            warm.append(time.time() - t0)
        warm.sort()
        median = warm[len(warm) // 2]
        print(f"warm {n_warm}x: {['%.3f' % t for t in warm]}, median={median*1000:.1f}ms",
              flush=True)
        return median
    except Exception as e:
        import traceback
        print(f"jax path FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()
        return None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", type=Path,
                   default=Path("Examples/compound_lens_zoo/mocks"))
    p.add_argument("--mock", type=int, default=3)
    p.add_argument("--n-warm", type=int, default=7)
    args = p.parse_args()

    print(f"Probe: multi-plane R5 likelihood timing on mock_{args.mock}",
          flush=True)
    print(f"Dataset root: {args.dataset_root}", flush=True)

    truths = json.loads((args.dataset_root /
                         f"truths_mock_{args.mock}.json").read_text())

    import autolens as al
    print(f"PyAutoLens: {al.__version__}", flush=True)

    dataset = load_dataset(args.dataset_root, args.mock)
    print(f"Dataset: shape={dataset.shape_native}, "
          f"pixels_in_mask={dataset.mask.pixels_in_mask}", flush=True)

    tracer = build_R5_tracer(truths)
    n_planes = len(set(g.redshift for g in tracer.galaxies))
    print(f"Tracer: {len(tracer.galaxies)} galaxies on {n_planes} unique redshift "
          f"planes (multi-plane: {n_planes > 2})", flush=True)
    print(flush=True)

    np_median = time_likelihood_numpy(dataset, tracer, n_warm=args.n_warm)
    jax_median = time_likelihood_jax(dataset, tracer, n_warm=args.n_warm)

    print(flush=True)
    print("=" * 60, flush=True)
    print(f"SUMMARY (mock_{args.mock}, multi-plane R5):", flush=True)
    print(f"  numpy median:     {np_median*1000:.1f} ms", flush=True)
    if jax_median is not None:
        ratio = np_median / jax_median
        print(f"  jax-GPU median:   {jax_median*1000:.1f} ms", flush=True)
        print(f"  speedup numpy/jax: {ratio:.2f}x  ({'JAX FASTER' if ratio > 1 else 'NUMPY FASTER'})",
              flush=True)
    else:
        print("  jax-GPU: FAILED (see traceback above)", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
