"""Great-circle distance between earthquake epicenters and sites."""

import numpy as np

from config import EARTH_RADIUS_KM


def calculate_distance(
    long_source: np.ndarray,
    lat_source: np.ndarray,
    long: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """Haversine distance in km (supports broadcasting)."""
    pi180 = np.pi / 180

    long_source = long_source * pi180
    lat_source = lat_source * pi180
    long = long * pi180
    lat = lat * pi180

    dlon = long - long_source
    dlat = lat - lat_source
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_source) * np.cos(lat) * np.sin(dlon / 2) ** 2
    )
    angles = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return EARTH_RADIUS_KM * angles
