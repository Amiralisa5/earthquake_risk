"""Ground motion prediction equations (GMPE) for magnitude-distance relationships."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.io import loadmat

from config import GMPE_MAT

logger = logging.getLogger(__name__)

_INTERP_EPS = np.float32(1e-12)
_REQUIRED_GMPE_KEYS = frozenset({"Acc", "M_max", "r_jb"})


def _interp_fraction(
    value: np.ndarray,
    v0: np.ndarray,
    v1: np.ndarray,
) -> np.ndarray:
    """
    Linear interpolation weight in [0, 1].

    Returns 0 when v0 == v1 (degenerate bracket). Clips to [0, 1] for
    values slightly outside the lookup table.
    """
    denom = v1 - v0
    t = np.divide(
        value - v0,
        denom,
        out=np.zeros_like(value, dtype=np.float32),
        where=np.abs(denom) > _INTERP_EPS,
    )
    return np.clip(t, 0.0, 1.0).astype(np.float32)


def bilinear_interpolate_pga(
    magnitude: np.ndarray,
    distance: np.ndarray,
    gmpe: dict[str, np.ndarray],
) -> np.ndarray:
    """
    Bilinear interpolation of PGA on a GMPE lookup table.

    PGA ≈ (1−t)(1−u)·z00 + t(1−u)·z10 + (1−t)u·z01 + t·u·z11
    """
    magnitude = np.asarray(magnitude, dtype=np.float32)
    distance = np.asarray(distance, dtype=np.float32)

    if magnitude.shape != distance.shape:
        raise ValueError(
            f"Magnitude shape {magnitude.shape} must match distance shape {distance.shape}"
        )

    acc = gmpe["Acc"]
    m_max = gmpe["M_max"]
    r_jb = gmpe["r_jb"]

    i1 = np.searchsorted(m_max, magnitude, side="right")
    i0 = np.clip(i1 - 1, 0, m_max.size - 2)
    i1 = i0 + 1

    m0 = m_max[i0]
    m1 = m_max[i1]
    t = _interp_fraction(magnitude, m0, m1)

    j1 = np.searchsorted(r_jb, distance, side="right")
    j0 = np.clip(j1 - 1, 0, r_jb.size - 2)
    j1 = j0 + 1

    r0 = r_jb[j0]
    r1 = r_jb[j1]
    u = _interp_fraction(distance, r0, r1)

    z00 = acc[i0, j0]
    z10 = acc[i1, j0]
    z01 = acc[i0, j1]
    z11 = acc[i1, j1]

    return (
        (1.0 - t) * (1.0 - u) * z00
        + t * (1.0 - u) * z10
        + (1.0 - t) * u * z01
        + t * u * z11
    )


def _validate_gmpe_arrays(acc: np.ndarray, m_max: np.ndarray, r_jb: np.ndarray) -> None:
    if acc.ndim != 2:
        raise ValueError(f"Expected 2D Acc array, got shape {acc.shape}")
    if acc.shape != (len(m_max), len(r_jb)):
        raise ValueError(
            f"Acc shape {acc.shape} does not match magnitude ({len(m_max)}) "
            f"and distance ({len(r_jb)}) axes"
        )
    if np.any(np.isnan(acc)):
        raise ValueError("GMPE Acc array contains NaN values")
    if np.any(acc < 0):
        logger.warning("GMPE contains negative acceleration values (unexpected)")


class GMPEInterpolator:
    """
    GMPE lookup table with bilinear interpolation for PGA.

    Pre-computed PGA as a function of magnitude and Joyner-Boore distance (r_jb).
    """

    def __init__(self) -> None:
        self._data: Optional[dict[str, np.ndarray]] = None

    def load(self, gmpe_path: Path | str = GMPE_MAT) -> dict[str, np.ndarray]:
        """Load GMPE from MATLAB .mat file (cached on this instance)."""
        if self._data is not None:
            logger.debug("Using cached GMPE data")
            return self._data

        gmpe_path = Path(gmpe_path)
        if not gmpe_path.is_file():
            raise FileNotFoundError(f"GMPE file not found: {gmpe_path}")

        logger.info("Loading GMPE data from %s", gmpe_path)

        try:
            gmpe_raw = loadmat(str(gmpe_path))
        except Exception as exc:
            raise ValueError(f"Failed to load GMPE MATLAB file: {exc}") from exc

        found_keys = set(gmpe_raw.keys()) & _REQUIRED_GMPE_KEYS
        if found_keys != _REQUIRED_GMPE_KEYS:
            missing = _REQUIRED_GMPE_KEYS - found_keys
            raise ValueError(f"GMPE file missing required arrays: {sorted(missing)}")

        acc = gmpe_raw["Acc"].astype(np.float32)
        m_max = gmpe_raw["M_max"].flatten().astype(np.float32)
        r_jb = gmpe_raw["r_jb"].flatten().astype(np.float32)
        _validate_gmpe_arrays(acc, m_max, r_jb)

        self._data = {"Acc": acc, "M_max": m_max, "r_jb": r_jb}

        logger.info(
            "GMPE loaded: %d magnitudes, %d distances, Acc range [%.4f, %.4f] g",
            len(m_max),
            len(r_jb),
            acc.min(),
            acc.max(),
        )
        return self._data

    def interpolate(
        self,
        magnitude: np.ndarray,
        distance: np.ndarray,
    ) -> np.ndarray:
        """Interpolate PGA (g) for magnitude and Joyner-Boore distance (km)."""
        if self._data is None:
            raise RuntimeError("GMPE data not loaded. Call load() first.")
        return bilinear_interpolate_pga(magnitude, distance, self._data)


_default_interpolator: Optional[GMPEInterpolator] = None


def _get_default_interpolator() -> GMPEInterpolator:
    global _default_interpolator
    if _default_interpolator is None:
        _default_interpolator = GMPEInterpolator()
    return _default_interpolator


def load_gmpe_data(gmpe_path: Path | str = GMPE_MAT) -> dict[str, np.ndarray]:
    """Load GMPE lookup table (cached module-level instance)."""
    return _get_default_interpolator().load(gmpe_path)


def calculate_pga(
    magnitude: np.ndarray,
    distance: np.ndarray,
    gmpe: Optional[dict[str, np.ndarray]] = None,
) -> np.ndarray:
    """
    Calculate Peak Ground Acceleration (PGA) from magnitude and distance.

    Args:
        magnitude: Earthquake magnitude(s).
        distance: Joyner-Boore distance(s) in km (same shape as magnitude).
        gmpe: Optional pre-loaded GMPE dict; uses cached default if None.

    Returns:
        PGA in gravitational acceleration units (g).
    """
    if gmpe is not None:
        return bilinear_interpolate_pga(magnitude, distance, gmpe)
    interpolator = _get_default_interpolator()
    interpolator.load()
    return interpolator.interpolate(magnitude, distance)
