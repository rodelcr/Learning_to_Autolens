"""
PyAutoLens v2026 SLaM pipeline helper.

This file bundles the canonical Source, Light, and Mass (SLaM) pipeline functions
from `autolens_workspace_latest/scripts/guides/modeling/slam_start_here.py` so
that tutorial notebooks can import them without requiring the legacy v2025 `slam`
package (which imports the renamed `AdaptiveBrightnessSplit` and breaks on v2026).

Each stage is exposed as a SimpleNamespace module so existing notebook code that
uses `source_lp.run(...)`, `source_pix.run_1(...)`, `source_pix.run_2(...)`,
`light_lp.run(...)`, and `mass_total.run(...)` continues to work.

Calling conventions match v2026 slam_start_here.py:
- Functions take `dataset` (not `analysis`); an `AnalysisImaging` is constructed internally
- MGE lens/source light profiles are the default
- `mesh_init`/`regularization_init` are passed as `af.Model` or classes
"""

import os
import types
from pathlib import Path

# Fail loudly at import time if PYAUTOFIT_TEST_MODE is set. With this flag
# enabled, every PyAutoFit search returns a random prior draw instead of
# sampling — every SLaM stage in this file would produce meaningless output.
# Mod 08's cached results_summary.json (chi²_red = 44.8, theta_E off by 15%)
# was a direct casualty of this leaking from autolens_workspace_latest's
# integration-testing shell into a production Jupyter session.
if os.environ.get("PYAUTOFIT_TEST_MODE"):
    raise RuntimeError(
        f"PYAUTOFIT_TEST_MODE={os.environ['PYAUTOFIT_TEST_MODE']!r} is set. "
        "PyAutoFit would skip sampling and return a random prior draw — "
        "every SLaM stage below would be meaningless. "
        "Unset it and restart the kernel:\n"
        "    unset PYAUTOFIT_TEST_MODE\n"
        "    (for notebooks: restart the kernel after unsetting)"
    )

import autofit as af
import autolens as al


def _source_lp_run(
    settings_search,
    dataset,
    mask_radius: float = 3.0,
    redshift_lens: float = 0.5,
    redshift_source: float = 1.0,
    lens_bulge=None,
    lens_disk=None,
    mass=None,
    shear=None,
    source_bulge=None,
    mass_centre=None,
    n_batch: int = 50,
):
    """SOURCE LP stage — fit MGE lens light + Isothermal mass + MGE source."""
    if lens_bulge is None:
        lens_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
        )
    if source_bulge is None:
        source_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            centre_prior_is_uniform=False,
        )
    if mass is None:
        mass = af.Model(al.mp.Isothermal)
    if shear is None:
        shear = af.Model(al.mp.ExternalShear)
    if mass_centre is not None:
        try:
            mass.centre = mass_centre
        except Exception:
            pass

    analysis = al.AnalysisImaging(dataset=dataset, use_jax=False)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=redshift_lens,
                bulge=lens_bulge,
                disk=lens_disk,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=redshift_source,
                bulge=source_bulge,
            ),
        ),
    )

    search = af.Nautilus(
        name="source_lp[1]",
        **settings_search.search_dict,
        n_live=100,
        n_batch=n_batch,
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def _source_pix_run_1(
    settings_search,
    dataset,
    source_lp_result,
    mesh_init=None,
    regularization_init=None,
    n_batch: int = 20,
):
    """SOURCE PIX 1 — initial pixelized source for building a high-quality adapt image."""
    if mesh_init is None:
        mesh_init = af.Model(al.mesh.RectangularAdaptDensity, shape=(28, 28))
    if regularization_init is None:
        regularization_init = al.reg.Adapt

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_lp_result
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_lp_result.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=False,
    )

    mass = al.util.chaining.mass_from(
        mass=source_lp_result.model.galaxies.lens.mass,
        mass_result=source_lp_result.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )
    shear = source_lp_result.model.galaxies.lens.shear

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=mass,
                shear=shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh_init,
                    regularization=regularization_init,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[1]",
        **settings_search.search_dict,
        n_live=100,
        n_batch=n_batch,
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def _source_pix_run_2(
    settings_search,
    dataset,
    source_lp_result,
    source_pix_result_1,
    mesh=None,
    regularization=None,
    n_batch: int = 20,
):
    """SOURCE PIX 2 — final adaptive pixelized source using pix_1 as adapt image."""
    if mesh is None:
        mesh = af.Model(al.mesh.RectangularAdaptImage, shape=(28, 28))
    if regularization is None:
        regularization = al.reg.Adapt

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_pix_result_1
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_pix_result_1.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=False,
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.lens.redshift,
                bulge=source_lp_result.instance.galaxies.lens.bulge,
                disk=source_lp_result.instance.galaxies.lens.disk,
                mass=source_pix_result_1.instance.galaxies.lens.mass,
                shear=source_pix_result_1.instance.galaxies.lens.shear,
            ),
            source=af.Model(
                al.Galaxy,
                redshift=source_lp_result.instance.galaxies.source.redshift,
                pixelization=af.Model(
                    al.Pixelization,
                    mesh=mesh,
                    regularization=regularization,
                ),
            ),
        ),
    )

    search = af.Nautilus(
        name="source_pix[2]",
        **settings_search.search_dict,
        n_live=50,
        n_batch=n_batch,
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def _light_lp_run(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    mask_radius: float = 3.0,
    lens_bulge=None,
    n_batch: int = 20,
):
    """LIGHT LP — refit lens light with mass+source fixed."""
    if lens_bulge is None:
        lens_bulge = al.model_util.mge_model_from(
            mask_radius=mask_radius,
            total_gaussians=20,
            gaussian_per_basis=2,
            centre_prior_is_uniform=True,
        )

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=False,
    )

    source = al.util.chaining.source_custom_model_from(
        result=source_result_for_source, source_is_model=False
    )

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=lens_bulge,
                disk=None,
                mass=source_result_for_lens.instance.galaxies.lens.mass,
                shear=source_result_for_lens.instance.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="light[1]",
        **settings_search.search_dict,
        n_live=100,
        n_batch=n_batch,
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


def _mass_total_run(
    settings_search,
    dataset,
    source_result_for_lens,
    source_result_for_source,
    light_result,
    mass=None,
    n_batch: int = 20,
):
    """MASS TOTAL — final mass model with everything else fixed."""
    if mass is None:
        mass = af.Model(al.mp.PowerLaw)

    galaxy_image_name_dict = al.galaxy_name_image_dict_via_result_from(
        result=source_result_for_lens
    )
    adapt_images = al.AdaptImages(galaxy_name_image_dict=galaxy_image_name_dict)

    analysis = al.AnalysisImaging(
        dataset=dataset,
        adapt_images=adapt_images,
        positions_likelihood_list=[
            source_result_for_source.positions_likelihood_from(
                factor=3.0, minimum_threshold=0.2
            )
        ],
        use_jax=False,
    )

    mass = al.util.chaining.mass_from(
        mass=mass,
        mass_result=source_result_for_lens.model.galaxies.lens.mass,
        unfix_mass_centre=True,
    )

    bulge = light_result.instance.galaxies.lens.bulge
    disk = light_result.instance.galaxies.lens.disk

    source = al.util.chaining.source_from(result=source_result_for_source)

    model = af.Collection(
        galaxies=af.Collection(
            lens=af.Model(
                al.Galaxy,
                redshift=source_result_for_lens.instance.galaxies.lens.redshift,
                bulge=bulge,
                disk=disk,
                mass=mass,
                shear=source_result_for_lens.model.galaxies.lens.shear,
            ),
            source=source,
        ),
    )

    search = af.Nautilus(
        name="mass_total[1]",
        **settings_search.search_dict,
        n_live=100,
        n_batch=n_batch,
    )
    return search.fit(model=model, analysis=analysis, **settings_search.fit_dict)


# Preserve the v2025-era module-with-.run() interface used by the existing notebooks.
source_lp = types.SimpleNamespace(run=_source_lp_run)
source_pix = types.SimpleNamespace(run_1=_source_pix_run_1, run_2=_source_pix_run_2)
light_lp = types.SimpleNamespace(run=_light_lp_run)
mass_total = types.SimpleNamespace(run=_mass_total_run)
