"""Vulnerability curves and earthquake-induced monetary loss calculations."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal, Optional

import numpy as np
import pandas as pd

from config import (
    VULNERABILITY_BUILDING_TYPES,
    VULNERABILITY_THRESHOLDS,
    VULNERABILITY_XLSX,
)

logger = logging.getLogger(__name__)

_INTERP_EPS = np.float32(1e-12)

BuildingType = Literal["SH", "SM", "SL", "CH", "CM", "CL", "MM", "ML"]
VALID_BUILDING_TYPES = frozenset(VULNERABILITY_BUILDING_TYPES)


def _interp_on_curve(
    pga_value: np.ndarray,
    pga_vals: np.ndarray,
    vul_vals: np.ndarray,
) -> np.ndarray:
    """Linear interpolation of vulnerability along the PGA axis."""
    id_x = np.searchsorted(pga_vals, pga_value) - 1
    id_x = np.clip(id_x, 0, len(pga_vals) - 2)

    x0 = pga_vals[id_x]
    x1 = pga_vals[id_x + 1]
    y0 = vul_vals[id_x]
    y1 = vul_vals[id_x + 1]

    denom = x1 - x0
    slope = np.divide(
        y1 - y0,
        denom,
        out=np.zeros_like(y0, dtype=np.float32),
        where=np.abs(denom) > _INTERP_EPS,
    )
    vul = y0 + slope * (pga_value - x0)
    return np.clip(vul, 0.0, 1.0).astype(np.float32)


def _validate_vulnerability_df(df: pd.DataFrame) -> None:
    if "PGA" not in df.columns:
        raise ValueError("Vulnerability file must contain a 'PGA' column")

    missing = VALID_BUILDING_TYPES - set(df.columns)
    if missing:
        raise ValueError(f"Vulnerability file missing building columns: {sorted(missing)}")

    pga = df["PGA"].values.astype(np.float64)
    if np.any(np.isnan(pga)):
        raise ValueError("PGA column contains NaN values")
    if not np.all(np.diff(pga) > 0):
        raise ValueError("PGA values must be strictly increasing")

    for btype in VALID_BUILDING_TYPES:
        if df[btype].isnull().any():
            raise ValueError(f"Building column '{btype}' contains null values")
        vul_col = df[btype].values
        if vul_col.min() < -1e-6 or vul_col.max() > 1.0 + 1e-6:
            raise ValueError(
                f"Vulnerability ratios for {btype} outside [0, 1]: "
                f"[{vul_col.min()}, {vul_col.max()}]"
            )


def interpolate_vulnerability(
    pga_value: np.ndarray,
    building_type: str,
    vul_data: dict[str, object],
) -> np.ndarray:
    """
    Interpolate damage ratio from PGA and apply building-type threshold cap.

    Below threshold: interpolated partial damage ratio.
    At or above threshold: capped at 1.0 (total damage).
    """
    if building_type not in VALID_BUILDING_TYPES:
        raise ValueError(
            f"Invalid building type: {building_type}. "
            f"Must be one of: {sorted(VALID_BUILDING_TYPES)}"
        )

    df = vul_data["df"]
    pga_vals = vul_data["pga_vals"]
    vul_vals = np.asarray(df[building_type].values, dtype=np.float32)

    vul = _interp_on_curve(
        np.asarray(pga_value, dtype=np.float32),
        pga_vals,
        vul_vals,
    )

    threshold = VULNERABILITY_THRESHOLDS[building_type]
    return np.where(vul >= threshold, np.float32(1.0), vul)


class VulnerabilityInterpolator:
    """
    Vulnerability curve interpolator for building damage assessment.

    Maps PGA (g) to damage ratios (0 = none, 1 = total) per building type code.
    """

    def __init__(self) -> None:
        self._data: Optional[dict[str, object]] = None

    def load(self, vul_path: Path | str = VULNERABILITY_XLSX) -> dict[str, object]:
        """Load vulnerability curves from Excel (cached on this instance)."""
        if self._data is not None:
            logger.debug("Using cached vulnerability data")
            return self._data

        vul_path = Path(vul_path)
        if not vul_path.is_file():
            raise FileNotFoundError(f"Vulnerability file not found: {vul_path}")

        logger.info("Loading vulnerability curves from %s", vul_path)

        try:
            df = pd.read_excel(vul_path)
        except Exception as exc:
            raise ValueError(f"Failed to load vulnerability Excel file: {exc}") from exc

        _validate_vulnerability_df(df)

        pga_vals = np.asarray(df["PGA"].values, dtype=np.float32)
        self._data = {"df": df, "pga_vals": pga_vals}

        logger.info(
            "Vulnerability loaded: %d PGA points, range [%.4f, %.4f] g",
            len(df),
            pga_vals.min(),
            pga_vals.max(),
        )
        return self._data

    def get_vulnerability(
        self,
        pga_value: np.ndarray,
        building_type: str,
        vul_data: Optional[dict[str, object]] = None,
    ) -> np.ndarray:
        """Interpolate vulnerability; uses cached load() data unless vul_data is passed."""
        data = vul_data if vul_data is not None else self._data
        if data is None:
            data = self.load()
        return interpolate_vulnerability(pga_value, building_type, data)


# Backward-compatible alias
VulnerabilityModel = VulnerabilityInterpolator

_default_interpolator: Optional[VulnerabilityInterpolator] = None


def _get_default_interpolator() -> VulnerabilityInterpolator:
    global _default_interpolator
    if _default_interpolator is None:
        _default_interpolator = VulnerabilityInterpolator()
    return _default_interpolator


def load_vulnerability_data(
    vul_path: Path | str = VULNERABILITY_XLSX,
) -> dict[str, object]:
    """Load vulnerability curves (cached module-level instance)."""
    return _get_default_interpolator().load(vul_path)


def calculate_vul(
    pga_value: np.ndarray,
    building_type: str,
    vul_data: Optional[dict[str, object]] = None,
) -> np.ndarray:
    """Calculate damage ratio from Peak Ground Acceleration."""
    if vul_data is not None:
        return interpolate_vulnerability(pga_value, building_type, vul_data)
    return _get_default_interpolator().get_vulnerability(pga_value, building_type)


def calculate_risk(
    vulnerability: np.ndarray,
    exposure_count: np.ndarray,
    unit_price: np.ndarray,
) -> np.ndarray:
    """
    Expected monetary loss per site.

    Loss = vulnerability × exposure_count × unit_price
    """
    return vulnerability * exposure_count * unit_price
