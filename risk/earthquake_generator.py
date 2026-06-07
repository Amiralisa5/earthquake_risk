"""Earthquake catalog generation (magnitude, count, epicenter location)."""

import random

import numpy as np
from shapely.geometry import Point, Polygon


def calculate_magnitude_distribution(
    m_min: float, m_max: float, b: float, num_points: int = 500
) -> tuple[np.ndarray, np.ndarray]:
    """Gutenberg-Richter truncated CDF: F(m) = (1 - e^{-β(m-m_min)}) / (1 - e^{-β(m_max-m_min)})
    where β = b·ln(10).  Returns magnitude array and corresponding CDF values."""
    beta = b
    m = np.linspace(m_min, m_max, num_points)
    tmp1 = 1 - np.exp(-beta * (m - m_min))
    tmp2 = 1 - np.exp(-beta * (m_max - m_min))
    f = tmp1 / tmp2
    return m, f


def generate_number_eq(a, b, m_min):
    lambda_ = 10 ** (a - b * m_min)
    return np.random.poisson(lambda_)


def generate_magnitude(k, b, m_min, m_max):
    m, f = calculate_magnitude_distribution(m_min, m_max, b)
    r = np.random.uniform(0, 1, size=k)
    indices = np.searchsorted(f, r)
    indices = np.clip(indices, 1, len(f) - 1)
    x0 = f[indices - 1]
    x1 = f[indices]
    y0 = m[indices - 1]
    y1 = m[indices]
    return y0 + (r - x0) * (y1 - y0) / (x1 - x0)


def generate_location(
    polygon_points: list[tuple[float, float]],
    max_retries: int = 10_000,
) -> tuple[float, float]:
    """Return a random (longitude, latitude) inside the fault polygon."""
    polygon = Polygon(polygon_points)
    bounds = polygon.bounds  # (minx, miny, maxx, maxy)
    for _ in range(max_retries):
        random_point = Point(
            random.uniform(bounds[0], bounds[2]),
            random.uniform(bounds[1], bounds[3]),
        )
        if polygon.contains(random_point):
            return random_point.x, random_point.y
    raise RuntimeError(
        f"Could not sample a point inside the fault polygon after {max_retries} attempts. "
        "Check that the polygon geometry is valid and non-degenerate."
    )


def generate_earthquakes(model_data) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate earthquake magnitudes and epicenters for all sources.

    Returns:
        results_mags, results_sources, long_eq, lat_eq
    """
    number_earthquake = generate_number_eq(
        model_data.source_a, model_data.source_b, model_data.source_m_min
    )

    results_mags = []
    results_sources = []
    long_eq = []
    lat_eq = []

    for idx in range(model_data.source_ids.size):
        k = int(number_earthquake[idx].item())
        if k <= 0:
            continue

        source_id = int(model_data.source_ids[idx].item())
        b = float(model_data.source_b[idx].item())
        mmin = float(model_data.source_m_min[idx].item())
        mmax = float(model_data.source_m_max[idx].item())
        mags = generate_magnitude(k, b, mmin, mmax)

        for mag in mags:
            results_mags.append(mag)
            results_sources.append(source_id)
            lon, lat = generate_location(model_data.sources_polygon[source_id - 1])
            long_eq.append(lon)
            lat_eq.append(lat)

    return (
        np.asarray(results_mags, dtype=np.float32),
        np.asarray(results_sources, dtype=np.int32),
        np.asarray(long_eq, dtype=np.float32),
        np.asarray(lat_eq, dtype=np.float32),
    )
