"""Tests for PGA bilinear interpolation (using a synthetic GMPE table)."""

import numpy as np
import pytest

from risk.pga_calculator import calculate_pga


def _make_gmpe(m_vals=None, r_vals=None, acc=None):
    """Build a minimal synthetic GMPE dict."""
    if m_vals is None:
        m_vals = np.array([5.0, 6.0, 7.0], dtype=np.float32)
    if r_vals is None:
        r_vals = np.array([10.0, 50.0, 100.0], dtype=np.float32)
    if acc is None:
        # acc[i, j] = magnitude * 0.1 / distance * 10 (synthetic)
        acc = np.outer(m_vals * 0.1, 1.0 / (r_vals / 10)).astype(np.float32)
    return {"Acc": acc, "M_max": m_vals, "r_jb": r_vals}


def test_exact_table_point():
    """PGA at an exact grid point should match the table value."""
    gmpe = _make_gmpe()
    mag = np.array([6.0], dtype=np.float32)
    dist = np.array([50.0], dtype=np.float32)
    result = calculate_pga(mag, dist, gmpe)
    expected = gmpe["Acc"][1, 1]
    assert float(result[0]) == pytest.approx(float(expected), rel=1e-4)


def test_output_shape():
    gmpe = _make_gmpe()
    n = 20
    mags = np.random.uniform(5.0, 7.0, n).astype(np.float32)
    dists = np.random.uniform(10.0, 100.0, n).astype(np.float32)
    result = calculate_pga(mags, dists, gmpe)
    assert result.shape == (n,)


def test_pga_non_negative():
    gmpe = _make_gmpe()
    mags = np.random.uniform(5.0, 7.0, 50).astype(np.float32)
    dists = np.random.uniform(10.0, 100.0, 50).astype(np.float32)
    result = calculate_pga(mags, dists, gmpe)
    assert np.all(result >= 0)


def test_pga_decreases_with_distance():
    """Holding magnitude fixed, PGA should decrease as distance increases."""
    gmpe = _make_gmpe()
    mag = np.array([6.0, 6.0], dtype=np.float32)
    dist_near = np.array([10.0], dtype=np.float32)
    dist_far = np.array([100.0], dtype=np.float32)
    pga_near = calculate_pga(np.array([6.0], dtype=np.float32), dist_near, gmpe)
    pga_far = calculate_pga(np.array([6.0], dtype=np.float32), dist_far, gmpe)
    assert float(pga_near[0]) > float(pga_far[0])
