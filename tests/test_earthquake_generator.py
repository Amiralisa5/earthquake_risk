"""Tests for earthquake catalog generation."""

import numpy as np
import pytest

from risk.earthquake_generator import (
    calculate_magnitude_distribution,
    generate_location,
    generate_magnitude,
    generate_number_eq,
)


# --- magnitude distribution ---

def test_cdf_starts_near_zero():
    _, f = calculate_magnitude_distribution(5.0, 7.5, 1.0)
    assert f[0] == pytest.approx(0.0, abs=1e-6)


def test_cdf_ends_at_one():
    _, f = calculate_magnitude_distribution(5.0, 7.5, 1.0)
    assert f[-1] == pytest.approx(1.0, abs=1e-6)


def test_cdf_is_monotonically_increasing():
    _, f = calculate_magnitude_distribution(5.0, 7.5, 1.0)
    assert np.all(np.diff(f) >= 0)


def test_magnitude_range_respected():
    m_min, m_max = 4.5, 7.0
    mags = generate_magnitude(500, b=1.0, m_min=m_min, m_max=m_max)
    assert mags.min() >= m_min - 1e-4
    assert mags.max() <= m_max + 1e-4


def test_magnitude_count():
    k = 100
    mags = generate_magnitude(k, b=1.0, m_min=5.0, m_max=7.5)
    assert len(mags) == k


# --- Poisson earthquake count ---

def test_generate_number_eq_non_negative():
    a = np.array([3.0, 2.5])
    b = np.array([1.0, 0.9])
    m_min = np.array([5.0, 5.0])
    counts = generate_number_eq(a, b, m_min)
    assert np.all(counts >= 0)


def test_generate_number_eq_shape():
    a = np.array([3.0, 2.5, 2.0])
    b = np.ones(3)
    m_min = np.full(3, 5.0)
    counts = generate_number_eq(a, b, m_min)
    assert counts.shape == (3,)


# --- location generation ---

def test_generate_location_inside_polygon():
    square = [(44.0, 30.0), (46.0, 30.0), (46.0, 32.0), (44.0, 32.0)]
    for _ in range(20):
        lon, lat = generate_location(square)
        assert 44.0 <= lon <= 46.0
        assert 30.0 <= lat <= 32.0


def test_generate_location_raises_on_degenerate_polygon():
    """A line (zero-area polygon) should exhaust retries and raise."""
    line = [(44.0, 30.0), (44.0, 30.0), (44.0, 30.0)]
    with pytest.raises(RuntimeError, match="Could not sample"):
        generate_location(line, max_retries=50)
