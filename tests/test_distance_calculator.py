"""Tests for the Haversine distance calculator."""

import numpy as np
import pytest

from risk.distance_calculator import calculate_distance


def test_same_point_is_zero():
    """Distance from a point to itself is zero."""
    d = calculate_distance(
        np.array([[45.0]]), np.array([[35.0]]),
        np.array([[45.0]]), np.array([[35.0]]),
    )
    assert d.item() == pytest.approx(0.0, abs=1e-3)


def test_known_distance():
    """Tehran (~51.4 E, 35.7 N) to Isfahan (~51.7 E, 32.7 N) ≈ 335 km."""
    d = calculate_distance(
        np.array([[51.4]]), np.array([[35.7]]),
        np.array([[51.7]]), np.array([[32.7]]),
    )
    assert d.item() == pytest.approx(335.0, rel=0.05)


def test_broadcasting_shape():
    """Output shape matches (n_earthquakes, n_sites)."""
    n_eq, n_site = 5, 10
    lon_src = np.random.uniform(44, 63, n_eq)[:, None]
    lat_src = np.random.uniform(25, 40, n_eq)[:, None]
    lon_site = np.random.uniform(44, 63, n_site)[None, :]
    lat_site = np.random.uniform(25, 40, n_site)[None, :]
    d = calculate_distance(lon_src, lat_src, lon_site, lat_site)
    assert d.shape == (n_eq, n_site)


def test_distances_non_negative():
    n_eq, n_site = 8, 12
    lon_src = np.random.uniform(44, 63, n_eq)[:, None]
    lat_src = np.random.uniform(25, 40, n_eq)[:, None]
    lon_site = np.random.uniform(44, 63, n_site)[None, :]
    lat_site = np.random.uniform(25, 40, n_site)[None, :]
    d = calculate_distance(lon_src, lat_src, lon_site, lat_site)
    assert np.all(d >= 0)


def test_symmetry():
    """Distance A→B equals distance B→A."""
    d_ab = calculate_distance(
        np.array([[50.0]]), np.array([[35.0]]),
        np.array([[55.0]]), np.array([[38.0]]),
    )
    d_ba = calculate_distance(
        np.array([[55.0]]), np.array([[38.0]]),
        np.array([[50.0]]), np.array([[35.0]]),
    )
    assert d_ab.item() == pytest.approx(d_ba.item(), rel=1e-5)
