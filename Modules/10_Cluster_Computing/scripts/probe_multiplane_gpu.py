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
import sys
import time
from pathlib import Path


def build_R5_truth_like_model(truths: dict):
    """Build an Isothermal-only R5 model with truth-anchored priors. Avoids
    the PowerLaw JAX bug so the vmap path can be exercised."""
    import autofit as af
    import autolens as al

    z_l1 = truths["redshifts"]["lens_primary"]
    z_l2 = truths["redshifts"]["lens_secondary"]
    z_s = truths["redshifts"]["source"]

    epls = [(i, kw) for i, (m, kw) in enumerate(
        zip(truths["lens_model_list"], truths["kwargs_lens"])) if m == "EPL"]
    primary, secondary = epls[0][1], epls[1][1]
    light = truths.get("kwargs_lens_light", [{}])[0]
    sources = truths.get("kwargs_source", [{}, {}])

    import math
    shear_kw = next((kw for m, kw in zip(truths["lens_model_list"],
                                          truths["kwargs_lens"])
                     if m == "SHEAR_GAMMA_PSI"), None)
    if shear_kw:
        gx, psi = shear_kw["gamma_ext"], shear_kw["psi_ext"]
        s_g1, s_g2 = gx * math.cos(2 * psi), gx * math.sin(2 * psi)
    else:
        s_g1, s_g2 = 0.0, 0.0

    # Primary lens light + Isothermal mass + ExternalShear
    bulge = af.Model(al.lp.Sersic)
    bulge.centre.centre_0 = af.GaussianPrior(mean=light.get("center_x", 0.0), sigma=0.05)
    bulge.centre.centre_1 = af.GaussianPrior(mean=light.get("center_y", 0.0), sigma=0.05)
    bulge.intensity        = af.LogUniformPrior(lower_limit=1e-6, upper_limit=1e6)
    bulge.effective_radius = af.GaussianPrior(mean=light.get("R_sersic", 1.0), sigma=0.3)
    bulge.sersic_index     = af.GaussianPrior(mean=light.get("n_sersic", 4.0), sigma=0.5)
    bulge.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=light.get("e1", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
    bulge.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=light.get("e2", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)

    mass_1 = af.Model(al.mp.Isothermal)  # NOT PowerLaw — avoids omega bug
    mass_1.centre.centre_0 = af.GaussianPrior(mean=primary.get("center_x", 0.0), sigma=0.05)
    mass_1.centre.centre_1 = af.GaussianPrior(mean=primary.get("center_y", 0.0), sigma=0.05)
    mass_1.einstein_radius = af.GaussianPrior(mean=primary.get("theta_E", 1.0), sigma=0.1)
    mass_1.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=primary.get("e1", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
    mass_1.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=primary.get("e2", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)

    shear = af.Model(al.mp.ExternalShear)
    shear.gamma_1 = af.GaussianPrior(mean=s_g1, sigma=0.05)
    shear.gamma_2 = af.GaussianPrior(mean=s_g2, sigma=0.05)

    lens_1 = af.Model(al.Galaxy, redshift=z_l1, bulge=bulge, mass=mass_1, shear=shear)

    mass_2 = af.Model(al.mp.Isothermal)
    mass_2.centre.centre_0 = af.GaussianPrior(mean=secondary.get("center_x", 0.0), sigma=0.05)
    mass_2.centre.centre_1 = af.GaussianPrior(mean=secondary.get("center_y", 0.0), sigma=0.05)
    mass_2.einstein_radius = af.GaussianPrior(mean=secondary.get("theta_E", 0.1), sigma=0.05)
    mass_2.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
        mean=secondary.get("e1", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
    mass_2.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
        mean=secondary.get("e2", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)

    lens_2 = af.Model(al.Galaxy, redshift=z_l2, mass=mass_2)

    def _src(seed):
        s = af.Model(al.lp.SersicCore)
        s.centre.centre_0 = af.GaussianPrior(mean=seed.get("center_x", 0.0), sigma=0.05)
        s.centre.centre_1 = af.GaussianPrior(mean=seed.get("center_y", 0.0), sigma=0.05)
        s.intensity        = af.LogUniformPrior(lower_limit=1e-3, upper_limit=1e3)
        s.effective_radius = af.GaussianPrior(mean=seed.get("R_sersic", 0.2), sigma=0.05)
        s.sersic_index     = af.GaussianPrior(mean=seed.get("n_sersic", 1.0), sigma=0.3)
        s.ell_comps.ell_comps_0 = af.TruncatedGaussianPrior(
            mean=seed.get("e1", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
        s.ell_comps.ell_comps_1 = af.TruncatedGaussianPrior(
            mean=seed.get("e2", 0.0), sigma=0.1, lower_limit=-1.0, upper_limit=1.0)
        return s

    source = af.Model(al.Galaxy, redshift=z_s,
                      bulge=_src(sources[0] if len(sources) > 0 else {}),
                      disk=_src(sources[1] if len(sources) > 1 else {}))

    return af.Collection(galaxies=af.Collection(
        lens_1=lens_1, lens_2=lens_2, source=source))


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


def time_via_analysis(dataset, tracer, truths, n_warm: int = 10,
                      use_jax: bool = False):
    """Time the actual code path Nautilus uses: AnalysisImaging.log_likelihood_function.

    Builds a model + instance from the same parameters as the tracer, so we
    exercise tracer_via_instance_from + fit_from (the full per-likelihood
    pipeline including position-likelihood penalties etc.).
    """
    import autolens as al
    import autofit as af

    backend = "use_jax=True" if use_jax else "use_jax=False"
    print(f"\n=== AnalysisImaging({backend}) — full Nautilus path ===",
          flush=True)

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=use_jax)

    # Build a model instance directly (no Nautilus involvement) — this is what
    # log_likelihood_function expects.
    instance = af.Collection(galaxies=af.Collection(
        **{f"g{i}": g for i, g in enumerate(tracer.galaxies)}))

    t0 = time.time()
    ll = float(analysis.log_likelihood_function(instance=instance))
    cold = time.time() - t0
    print(f"first call (cold+jit if jax): ll={ll:.2f}, time={cold:.3f}s",
          flush=True)

    warm = []
    for _ in range(n_warm):
        t0 = time.time()
        _ = float(analysis.log_likelihood_function(instance=instance))
        warm.append(time.time() - t0)
    warm.sort()
    median = warm[len(warm) // 2]
    print(f"warm {n_warm}x: {['%.3f' % t for t in warm]}, median={median*1000:.1f}ms",
          flush=True)
    return median, cold


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

    # Now exercise the ACTUAL code path Nautilus uses (AnalysisImaging.log_likelihood_function).
    # The single-call probe above tests the inner FitImaging directly, which
    # may not amortize JAX JIT compilation across calls. The Analysis-level
    # path is what Nautilus calls 100k+ times during a fit, so its warm
    # median is the real-world throughput estimator.
    np_anal_median, np_anal_cold = time_via_analysis(
        dataset, tracer, truths, n_warm=args.n_warm, use_jax=False)
    try:
        jax_anal_median, jax_anal_cold = time_via_analysis(
            dataset, tracer, truths, n_warm=args.n_warm, use_jax=True)
    except Exception as e:
        import traceback
        print(f"AnalysisImaging(use_jax=True) FAILED: {type(e).__name__}: {e}",
              flush=True)
        traceback.print_exc()
        jax_anal_median, jax_anal_cold = None, None

    # Now the BATCHED vmap path — what Nautilus actually uses with use_jax_vmap=True.
    # Fitness._vmap = jax.vmap(jax.jit(self.call)) so a batch of N proposals
    # evaluates as ONE GPU op. Per-call cost = batch_total / N.
    print(flush=True)
    print("=== Fitness._vmap (JAX vmap'd batch — what Nautilus uses) ===",
          flush=True)
    try:
        import autofit as af
        import autolens as al
        import numpy as np
        from autofit.non_linear.fitness import Fitness

        analysis_jax = al.AnalysisImaging(dataset=dataset, use_jax=True)

        # Build a model that matches the tracer (so we can sample param vectors)
        model = build_R5_truth_like_model(truths)
        n_params = model.prior_count
        print(f"  model has {n_params} free parameters", flush=True)

        fitness = Fitness(
            model=model,
            analysis=analysis_jax,
            paths=None,
            fom_is_log_likelihood=True,
            use_jax_vmap=True,
        )

        # Sample BATCH_SIZE parameter vectors uniformly in unit-cube (priors map them)
        # CRITICAL: JAX dispatch is async — must block on the result to measure
        # actual compute, not just kernel-launch latency.
        import jax
        for batch_size in [1, 16, 64, 256]:
            rng = np.random.default_rng(seed=42)
            batch = rng.uniform(0.01, 0.99, size=(batch_size, n_params))
            t0 = time.time()
            try:
                result = fitness._call(batch)
                # Force completion (jax.block_until_ready handles arrays + scalars)
                jax.block_until_ready(result)
                cold = time.time() - t0
            except Exception as e:
                print(f"  batch_size={batch_size}: FAILED at cold call: {e}",
                      flush=True)
                continue
            warm_times = []
            for _ in range(min(5, args.n_warm)):
                t0 = time.time()
                result = fitness._call(batch)
                jax.block_until_ready(result)
                warm_times.append(time.time() - t0)
            warm_times.sort()
            warm_med = warm_times[len(warm_times) // 2]
            per_call = warm_med / batch_size
            speedup = (np_anal_median / per_call) if per_call > 0 else float("inf")
            print(f"  batch_size={batch_size}: cold {cold:.2f}s, "
                  f"warm median {warm_med*1000:.1f}ms total → "
                  f"{per_call*1000:.2f}ms/call  ({speedup:.1f}x vs numpy)",
                  flush=True)
    except Exception as e:
        import traceback
        print(f"Fitness vmap probe FAILED: {type(e).__name__}: {e}", flush=True)
        traceback.print_exc()

    print(flush=True)
    print("=" * 70, flush=True)
    print(f"SUMMARY (mock_{args.mock}, multi-plane R5):", flush=True)
    print(f"  -- FitImaging direct calls --", flush=True)
    print(f"  numpy median:        {np_median*1000:.1f} ms", flush=True)
    if jax_median is not None:
        ratio = np_median / jax_median
        print(f"  jax-GPU median:      {jax_median*1000:.1f} ms  ({ratio:.2f}x vs numpy)",
              flush=True)
    print(f"  -- AnalysisImaging.log_likelihood_function (real Nautilus path) --",
          flush=True)
    print(f"  numpy median:        {np_anal_median*1000:.1f} ms  (cold {np_anal_cold:.1f}s)",
          flush=True)
    if jax_anal_median is not None:
        ratio = np_anal_median / jax_anal_median
        print(f"  jax-GPU median:      {jax_anal_median*1000:.1f} ms  (cold {jax_anal_cold:.1f}s, {ratio:.2f}x vs numpy)",
              flush=True)
        # Estimate full-fit wall time at 100k likelihood calls
        np_h = 100_000 * np_anal_median / 3600
        jx_h = (100_000 * jax_anal_median + jax_anal_cold) / 3600
        print(f"  100k-call estimate:  numpy {np_h:.2f} h  vs  jax-GPU {jx_h:.2f} h",
              flush=True)
    else:
        print("  jax-GPU AnalysisImaging: FAILED", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
