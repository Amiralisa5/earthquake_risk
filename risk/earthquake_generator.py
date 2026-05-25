"""
Earthquake catalog generation from seismic sources.

Generates synthetic catalogs using:
1. Poisson counts from the Gutenberg-Richter relation
2. Magnitude sampling from the truncated exponential GR CDF
3. Uniform epicenter locations within fault polygons (triangulation-based)
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from shapely.geometry import Polygon
from shapely.ops import triangulate

logger = logging.getLogger(__name__)

_GR_EPS = np.float32(1e-12)

# Cached triangle meshes: polygon key -> (vertices [n,3,2], weights [n])
_POLYGON_SAMPLER_CACHE: dict[tuple[tuple[float, float], ...], tuple[np.ndarray, np.ndarray]] = {}


def calculate_magnitude_distribution(
    m_min: float,
    m_max: float,
    b: float,
    num_points: int = 500,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Truncated Gutenberg-Richter cumulative distribution for magnitude sampling.

    P(M) = [1 - exp(-β(M - M_min))] / [1 - exp(-β(M_max - M_min))],  β = b·ln(10)
    """
    beta = b * np.log(10.0)
    magnitudes = np.linspace(m_min, m_max, num_points, dtype=np.float32)

    numerator = 1.0 - np.exp(-beta * (magnitudes - m_min))
    denominator = 1.0 - np.exp(-beta * (m_max - m_min))
    probabilities = numerator / max(denominator, _GR_EPS)
    probabilities = np.clip(probabilities, 0.0, 1.0).astype(np.float32)

    return magnitudes, probabilities


def generate_number_eq(
    a_values: np.ndarray,
    b_values: np.ndarray,
    m_min_values: np.ndarray,
) -> np.ndarray:
    """
    Sample earthquake counts per source: N ~ Poisson(λ),  λ = 10^(a - b·M_min).
    """
    a_values = np.asarray(a_values, dtype=np.float64)
    b_values = np.asarray(b_values, dtype=np.float64)
    m_min_values = np.asarray(m_min_values, dtype=np.float64)

    lambda_values = np.power(10.0, a_values - b_values * m_min_values)
    lambda_values = np.clip(lambda_values, 0.0, 1e10)
    return np.random.poisson(lambda_values)


def generate_magnitude(
    k: int,
    b: float,
    m_min: float,
    m_max: float,
) -> np.ndarray:
    """Sample k magnitudes via inverse transform on the GR CDF."""
    if k <= 0:
        return np.empty(0, dtype=np.float32)

    magnitudes, probs = calculate_magnitude_distribution(m_min, m_max, b)
    uniform_random = np.random.uniform(0.0, 1.0, size=k)

    indices = np.searchsorted(probs, uniform_random)
    indices = np.clip(indices, 1, len(probs) - 1)

    idx_below = indices - 1
    mag_below = magnitudes[idx_below]
    mag_above = magnitudes[indices]
    prob_below = probs[idx_below]
    prob_above = probs[indices]

    prob_range = prob_above - prob_below
    weight = np.divide(
        uniform_random - prob_below,
        prob_range,
        out=np.zeros_like(uniform_random, dtype=np.float32),
        where=prob_range > _GR_EPS,
    )
    weight = np.clip(weight, 0.0, 1.0)

    return (mag_below + weight * (mag_above - mag_below)).astype(np.float32)


def _polygon_cache_key(polygon_points: list) -> tuple[tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in polygon_points)


def _build_triangle_sampler(polygon_points: list) -> tuple[np.ndarray, np.ndarray]:
    polygon = Polygon(polygon_points)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    if polygon.is_empty or polygon.area <= 0:
        raise ValueError("Fault polygon is empty or has zero area")

    triangles = [
        tri
        for tri in triangulate(polygon)
        if tri.area > 0 and polygon.contains(tri.representative_point())
    ]
    if not triangles:
        triangles = [polygon]

    verts = np.array(
        [list(t.exterior.coords)[:3] for t in triangles],
        dtype=np.float64,
    )
    weights = np.array([t.area for t in triangles], dtype=np.float64)
    weights /= weights.sum()
    return verts, weights


def _get_triangle_sampler(polygon_points: list) -> tuple[np.ndarray, np.ndarray]:
    key = _polygon_cache_key(polygon_points)
    cached = _POLYGON_SAMPLER_CACHE.get(key)
    if cached is None:
        cached = _build_triangle_sampler(polygon_points)
        _POLYGON_SAMPLER_CACHE[key] = cached
    return cached


class PolygonPointSampler:
    """
    Uniform random point sampling inside a polygon via triangulation.

    Area-weighted triangle selection + barycentric sampling (O(1) per point after
    cache warm-up). Does not use rejection sampling.
    """

    def __init__(self, polygon_points: list):
        self.polygon_points = polygon_points
        self._verts, self._weights = _get_triangle_sampler(polygon_points)
        area = Polygon(polygon_points).area
        if area < 0.01:
            logger.warning("Small fault polygon area (%.6f deg²); check geometry", area)

    def sample(self, n: int = 1) -> tuple[np.ndarray, np.ndarray]:
        """Return (longitudes, latitudes) arrays of shape (n,)."""
        if n <= 0:
            return (
                np.empty(0, dtype=np.float32),
                np.empty(0, dtype=np.float32),
            )

        tri_idx = np.random.choice(len(self._weights), size=n, p=self._weights)
        r1 = np.random.random(n)
        r2 = np.random.random(n)
        overflow = r1 + r2 > 1.0
        r1[overflow] = 1.0 - r1[overflow]
        r2[overflow] = 1.0 - r2[overflow]

        triangles = self._verts[tri_idx]
        w0 = 1.0 - r1 - r2
        points = (
            w0[:, None] * triangles[:, 0]
            + r1[:, None] * triangles[:, 1]
            + r2[:, None] * triangles[:, 2]
        )
        return points[:, 0].astype(np.float32), points[:, 1].astype(np.float32)


def generate_location(polygon_points: list) -> tuple[float, float]:
    """Generate one random (longitude, latitude) inside a fault polygon."""
    lons, lats = PolygonPointSampler(polygon_points).sample(n=1)
    return float(lons[0]), float(lats[0])


def generate_earthquakes(
    model_data,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Generate earthquake catalog from all seismic sources.

    Returns:
        magnitudes, source_ids, longitudes, latitudes (each shape (n_earthquakes,))
    """
    logger.info("Generating earthquake catalog...")

    earthquake_counts = generate_number_eq(
        model_data.source_a,
        model_data.source_b,
        model_data.source_m_min,
    )
    total = int(np.sum(earthquake_counts))
    logger.info(
        "Generating %d earthquakes from %d sources",
        total,
        len(model_data.source_ids),
    )

    if total == 0:
        return (
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    magnitudes = np.empty(total, dtype=np.float32)
    source_ids = np.empty(total, dtype=np.int32)
    longitudes = np.empty(total, dtype=np.float32)
    latitudes = np.empty(total, dtype=np.float32)

    eq_index = 0
    sampler_cache: dict[int, PolygonPointSampler] = {}

    for source_idx in range(len(model_data.source_ids)):
        k = int(earthquake_counts[source_idx])
        if k <= 0:
            continue

        source_id = int(model_data.source_ids[source_idx])
        b_value = float(model_data.source_b[source_idx])
        m_min = float(model_data.source_m_min[source_idx])
        m_max = float(model_data.source_m_max[source_idx])

        mags = generate_magnitude(k, b_value, m_min, m_max)

        polygon_idx = source_id - 1
        if polygon_idx not in sampler_cache:
            sampler_cache[polygon_idx] = PolygonPointSampler(
                model_data.sources_polygon[polygon_idx]
            )
        lons, lats = sampler_cache[polygon_idx].sample(n=k)

        magnitudes[eq_index : eq_index + k] = mags
        source_ids[eq_index : eq_index + k] = source_id
        longitudes[eq_index : eq_index + k] = lons
        latitudes[eq_index : eq_index + k] = lats
        eq_index += k

    logger.info(
        "Earthquake catalog generated: %d events, M in [%.2f, %.2f], %d sources",
        total,
        magnitudes.min(),
        magnitudes.max(),
        len(np.unique(source_ids)),
    )

    return magnitudes, source_ids, longitudes, latitudes
