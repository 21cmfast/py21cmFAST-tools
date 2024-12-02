"""Test cases for the __main__ module."""

import numpy as np

from py21cmfast_tools import calculate_ps


def test_calculate_ps():
    rng = np.random.default_rng()
    test_lc = rng.random((100, 100, 1000))
    test_redshifts = np.logspace(np.log10(5), np.log10(30), 1000)
    zs = [5.0, 6.0, 10.0, 27.0]

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
