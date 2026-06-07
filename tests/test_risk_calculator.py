"""Tests for vulnerability curves and loss calculation."""

import numpy as np
import pytest

from risk.risk_calculator import calculate_risk, calculate_vul


def _make_vul_data():
    """Synthetic vulnerability data: linear ramp from PGA 0→1 gives vul 0→1."""
    import pandas as pd

    pga = np.linspace(0.0, 1.0, 11)
    df = pd.DataFrame({"PGA": pga, "SH": pga, "CL": pga * 0.5})
    return {"df": df, "pga_vals": pga}


def _make_thresholds():
    return {"SH": 0.9, "CL": 0.9}


def test_zero_pga_gives_zero_vulnerability():
    vul_data = _make_vul_data()
    thresholds = _make_thresholds()
    pga = np.zeros(5, dtype=np.float32)
    result = calculate_vul(pga, "SH", vul_data, thresholds)
    assert np.all(result == 0.0)


def test_high_pga_clamped_to_one():
    """PGA above the threshold should return vulnerability = 1.0."""
    vul_data = _make_vul_data()
    thresholds = _make_thresholds()
    pga = np.ones(5, dtype=np.float32)  # at maximum; vulnerability = 1.0 >= threshold 0.9
    result = calculate_vul(pga, "SH", vul_data, thresholds)
    assert np.all(result == 1.0)


def test_vulnerability_in_range():
    vul_data = _make_vul_data()
    thresholds = _make_thresholds()
    pga = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    result = calculate_vul(pga, "SH", vul_data, thresholds)
    assert np.all(result >= 0.0)
    assert np.all(result <= 1.0)


def test_calculate_risk_formula():
    """Loss = vul × area × count × price."""
    vul = np.array([0.5, 0.5], dtype=np.float32)
    count = np.array([100, 200], dtype=np.int32)
    price = np.array([1_000_000.0, 500_000.0], dtype=np.float32)
    area = np.array([200.0, 150.0], dtype=np.float32)
    result = calculate_risk(vul, count, price, area)
    expected = vul * area * count * price
    np.testing.assert_allclose(result, expected, rtol=1e-5)


def test_calculate_risk_zero_vul():
    vul = np.zeros(10, dtype=np.float32)
    count = np.ones(10, dtype=np.int32) * 100
    price = np.ones(10, dtype=np.float32) * 1e6
    area = np.ones(10, dtype=np.float32) * 200.0
    result = calculate_risk(vul, count, price, area)
    assert np.all(result == 0.0)


def test_calculate_risk_non_negative():
    rng = np.random.default_rng(42)
    vul = rng.uniform(0, 1, 50).astype(np.float32)
    count = rng.integers(1, 500, 50).astype(np.int32)
    price = rng.uniform(1e5, 1e7, 50).astype(np.float32)
    area = rng.uniform(50, 500, 50).astype(np.float32)
    result = calculate_risk(vul, count, price, area)
    assert np.all(result >= 0)
