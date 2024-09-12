"""Test cases for the __main__ module."""

import numpy as np
from py21cmfast_tools import calculate_ps, ps_2d21d


def test_calculate_ps():
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        box_side_shape=100,
        zs=zs,
        calc_2d=False,
        calc_1d=True,
        calc_global=True,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        zs=zs,
        calc_1d=True,
        calc_global=True,
        interp=True,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        zs=zs,
        calc_1d=True,
        calc_global=True,
        mu=0.5,
    )

    calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        zs=zs,
        calc_1d=True,
        calc_global=True,
        interp=True,
        mu=0.5,
    )


def test_calculate_ps_w_var():
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [6.0, 10.0, 27.0]

    out = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        box_side_shape=100,
        zs=zs,
        calc_2d=False,
        calc_1d=True,
        calc_global=True,
        get_variance=True,
        postprocess=False,
    )
    out["var_1D"]
    out = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        box_side_shape=100,
        zs=zs,
        calc_2d=True,
        calc_1d=True,
        calc_global=True,
        get_variance=True,
        postprocess=False,
    )
    out["full_var_2D"]
    out["var_1D"]
    out = calculate_ps(
        test_lc,
        test_redshifts,
        box_length=200,
        zs=zs,
        calc_1d=True,
        calc_2d=True,
        calc_global=True,
        get_variance=True,
        postprocess=True,
    )
    out["final_var_2D"]
    with np.testing.assert_raises(NotImplementedError):
        calculate_ps(
            test_lc,
            test_redshifts,
            box_length=200,
            zs=zs,
            calc_1d=True,
            calc_global=True,
            get_variance=True,
            postprocess=False,
            interp="linear",
        )

    with np.testing.assert_raises(ValueError):
        calculate_ps(
            test_lc,
            test_redshifts,
            box_length=200,
            zs=[4.0],  # outside test_redshifts
            calc_1d=True,
            calc_global=True,
            get_variance=True,
            postprocess=True,
        )
    with np.testing.assert_raises(ValueError):
        calculate_ps(
            test_lc,
            test_redshifts,
            box_length=200,
            zs=[50.0],  # outside test_redshifts
            calc_1d=True,
            calc_global=True,
            get_variance=True,
            postprocess=True,
        )


def test_ps_avg():
    rng = np.random.default_rng()
    ps_2d = rng.random((32, 32))
    x = np.linspace(0, 1, 32)
    ps, k, sws = ps_2d21d(ps_2d, x, x, nbins=16)
    assert ps.shape == (16,)
    assert k.shape == (16,)
    assert sws.shape == (16,)
    kpar_mesh, kperp_mesh = np.meshgrid(x, x)
    theta = np.arctan(kperp_mesh / kpar_mesh)
    mu_mesh = np.cos(theta)
    mask = mu_mesh >= 0.9
    ps_2d[mask] = 1000
    ps, k, sws = ps_2d21d(ps_2d, x, x, nbins=32, interp=True, mu=0.98)
    assert np.nanmean(ps[-20:]) == 1000.0
