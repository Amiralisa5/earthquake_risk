"""Great-circle distance between earthquake epicenters and sites."""

import numpy as np

# WGS84 mean Earth radius (km)
EARTH_RADIUS_KM = 6378.137


def calculate_distance(
    long_source: np.ndarray,
    lat_source: np.ndarray,
    long: np.ndarray,
    lat: np.ndarray,
) -> np.ndarray:
    """
    Calculate great-circle (Haversine) distances between source and site locations.

    Supports numpy broadcasting for efficient batch calculations.

    Haversine formula on a sphere:
        a = sin²(Δlat/2) + cos(lat1)·cos(lat2)·sin²(Δlon/2)
        c = 2·atan2(√a, √(1−a))
        d = R·c

    Args:
        long_source: Earthquake longitude(s) in decimal degrees.
        lat_source: Earthquake latitude(s) in decimal degrees.
        long: Site longitude(s) in decimal degrees.
        lat: Site latitude(s) in decimal degrees.

    Returns:
        Distances in kilometers. With shapes (n_eq, 1) and (1, n_sites),
        result shape is (n_eq, n_sites).

    Example:
        >>> lon_src = np.array([[10.0, 20.0]])   # shape (1, 2)
        >>> lat_src = np.array([[35.0, 40.0]])
        >>> lon_site = np.array([[15.0], [25.0], [35.0]])  # shape (3, 1)
        >>> lat_site = np.array([[37.0], [42.0], [45.0]])
        >>> dist = calculate_distance(lon_src, lat_src, lon_site, lat_site)
        >>> dist.shape
        (3, 2)

    Notes:
        - All inputs must be in decimal degrees.
        - Earth radius: 6378.137 km (WGS84 mean radius).

    References:
        - https://en.wikipedia.org/wiki/Haversine_formula
    """
    pi180 = np.pi / 180.0
    long_source_rad = long_source * pi180
    lat_source_rad = lat_source * pi180
    long_rad = long * pi180
    lat_rad = lat * pi180

    dlon = long_rad - long_source_rad
    dlat = lat_rad - lat_source_rad

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat_source_rad) * np.cos(lat_rad) * np.sin(dlon / 2.0) ** 2
    )
    a = np.clip(a, 0.0, 1.0)

    angles = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * angles
