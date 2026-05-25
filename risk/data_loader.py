"""Load exposure sites, fault sources, and derived numpy arrays."""

from dataclasses import dataclass
from pathlib import Path
import logging
from typing import Final

import numpy as np
import pandas as pd

from config import EXPOSURE_CSV, FAULTS_XLSX

logger = logging.getLogger(__name__)

# Expected schema
EXPOSURE_COLUMNS: Final[list[str]] = [
    "Site_ID",
    "Longitude",
    "Latitude",
    "SH",
    "SM",
    "SL",
    "CH",
    "CM",
    "CL",
    "MM",
    "ML",
    "SP",
    "CP",
    "MP",
    "Province",
]

SITE_NUMERIC_COLUMNS: Final[list[str]] = [
    "Site_ID",
    "Longitude",
    "Latitude",
    "SH",
    "SM",
    "SL",
    "CH",
    "CM",
    "CL",
    "MM",
    "ML",
    "SP",
    "CP",
    "MP",
]

EXPOSURE_COUNT_COLUMNS: Final[list[str]] = [
    "SH",
    "SM",
    "SL",
    "CH",
    "CM",
    "CL",
    "MM",
    "ML",
    "SP",
    "CP",
    "MP",
]

FAULT_COLUMNS: Final[list[str]] = [
    "Fault Number",
    "a_value",
    "b_value",
    "min_mag",
    "max_mag",
    "Geometry",
]

FAULT_NUMERIC_COLUMNS: Final[list[str]] = [
    "Fault Number",
    "a_value",
    "b_value",
    "min_mag",
    "max_mag",
]

# Iran bounding box (warn only — exposure may include border buffers)
IRAN_LON_RANGE: Final[tuple[float, float]] = (44.0, 64.0)
IRAN_LAT_RANGE: Final[tuple[float, float]] = (25.0, 40.0)


class DataValidationError(ValueError):
    """Raised when input data fails schema or quality checks."""


def validate_exposure_data(df: pd.DataFrame) -> None:
    """Validate exposure inventory CSV contents."""
    if df.empty:
        raise DataValidationError("Exposure data is empty")

    missing_cols = [col for col in EXPOSURE_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns in exposure data: {missing_cols}. "
            f"Found: {list(df.columns)}"
        )

    critical_cols = ["Site_ID", "Longitude", "Latitude", "Province"]
    null_summary = df[critical_cols].isnull().sum()
    if null_summary.any():
        raise DataValidationError(
            f"Found null values in critical columns:\n{null_summary[null_summary > 0]}"
        )

    numeric_nulls = df[SITE_NUMERIC_COLUMNS].isnull().sum()
    if numeric_nulls.any():
        raise DataValidationError(
            f"Found null values in numeric site columns:\n"
            f"{numeric_nulls[numeric_nulls > 0]}"
        )

    if df["Site_ID"].duplicated().any():
        dupes = df.loc[df["Site_ID"].duplicated(), "Site_ID"].unique()[:10]
        raise DataValidationError(
            f"Duplicate Site_ID values found (showing up to 10): {dupes}"
        )

    if not np.issubdtype(df["Site_ID"].dtype, np.integer):
        if not np.allclose(df["Site_ID"], df["Site_ID"].astype(int)):
            raise DataValidationError("Site_ID must be integer-valued")
    if (df["Site_ID"] <= 0).any():
        raise DataValidationError("Site_ID must be positive")

    if not df["Longitude"].between(-180, 180, inclusive="both").all():
        raise DataValidationError(
            f"Invalid longitude values. "
            f"Range: [{df['Longitude'].min()}, {df['Longitude'].max()}]"
        )
    if not df["Latitude"].between(-90, 90, inclusive="both").all():
        raise DataValidationError(
            f"Invalid latitude values. "
            f"Range: [{df['Latitude'].min()}, {df['Latitude'].max()}]"
        )

    lon_outside = ~df["Longitude"].between(*IRAN_LON_RANGE)
    lat_outside = ~df["Latitude"].between(*IRAN_LAT_RANGE)
    if lon_outside.any() or lat_outside.any():
        logger.warning(
            "%d sites outside Iran lon [%s, %s]; %d outside lat [%s, %s]",
            lon_outside.sum(),
            *IRAN_LON_RANGE,
            lat_outside.sum(),
            *IRAN_LAT_RANGE,
        )

    if (df[EXPOSURE_COUNT_COLUMNS] < 0).any().any():
        raise DataValidationError("Found negative exposure values (expected non-negative)")

    if (df["Province"].astype(str).str.strip() == "").any():
        raise DataValidationError("Found empty Province values")

    logger.info(
        "Exposure validation passed: %d sites, %d provinces",
        len(df),
        df["Province"].nunique(),
    )


def validate_fault_data(df: pd.DataFrame) -> None:
    """Validate fault source Excel contents."""
    if df.empty:
        raise DataValidationError("Fault data is empty")

    missing_cols = [col for col in FAULT_COLUMNS if col not in df.columns]
    if missing_cols:
        raise DataValidationError(
            f"Missing required columns in fault data: {missing_cols}"
        )

    null_summary = df[FAULT_COLUMNS].isnull().sum()
    if null_summary.any():
        raise DataValidationError(
            f"Found null values in fault columns:\n{null_summary[null_summary > 0]}"
        )

    if df["Fault Number"].duplicated().any():
        dupes = df.loc[df["Fault Number"].duplicated(), "Fault Number"].unique()[:10]
        raise DataValidationError(
            f"Duplicate Fault Number values found (showing up to 10): {dupes}"
        )

    fault_ids = df["Fault Number"].astype(int)
    expected = np.arange(1, len(df) + 1)
    if not np.array_equal(np.sort(fault_ids.values), expected):
        raise DataValidationError(
            f"Fault Number must be unique integers from 1 to {len(df)}. "
            f"Got range [{fault_ids.min()}, {fault_ids.max()}]"
        )
    if not np.array_equal(fault_ids.values, expected):
        logger.warning(
            "Fault rows are not ordered by Fault Number; polygons are indexed by row order "
            "(source_id - 1). Sorting faults by Fault Number before use is recommended."
        )

    if (df["min_mag"] >= df["max_mag"]).any():
        bad = df.loc[df["min_mag"] >= df["max_mag"], ["Fault Number", "min_mag", "max_mag"]]
        raise DataValidationError(
            f"Found min_mag >= max_mag:\n{bad.head().to_string()}"
        )

    if (df["b_value"] <= 0).any():
        raise DataValidationError("b_value must be positive for magnitude distribution")

    if ((df["b_value"] < 0.3) | (df["b_value"] > 3.0)).any():
        logger.warning(
            "b-values outside typical range [0.3, 3.0]: %s",
            df.loc[(df["b_value"] < 0.3) | (df["b_value"] > 3.0), "b_value"].tolist(),
        )

    if (df["a_value"] <= 0).any():
        raise DataValidationError("a_value must be positive")

    logger.info("Fault validation passed: %d fault sources", len(df))


@dataclass
class ModelData:
    """Data container for earthquake risk model."""

    exposure_df: pd.DataFrame
    provinces: np.ndarray
    site_ids: np.ndarray
    site_long: np.ndarray
    site_lat: np.ndarray
    site_sh: np.ndarray
    site_sm: np.ndarray
    site_sl: np.ndarray
    site_ch: np.ndarray
    site_cm: np.ndarray
    site_cl: np.ndarray
    site_mm: np.ndarray
    site_ml: np.ndarray
    site_sp: np.ndarray
    site_cp: np.ndarray
    site_mp: np.ndarray
    site_to_province: dict
    source_ids: np.ndarray
    source_a: np.ndarray
    source_b: np.ndarray
    source_m_min: np.ndarray
    source_m_max: np.ndarray
    sources_polygon: list


def load_polygon_source(faults_df: pd.DataFrame) -> list:
    """
    Parse WKT LINESTRING geometry into coordinate tuples.

    Polygon list index i corresponds to Fault Number i + 1 (row order).
    """
    sources_polygon = []
    for idx, fault in faults_df.iterrows():
        try:
            geometry = str(fault["Geometry"]).strip()
            if not geometry.upper().startswith("LINESTRING"):
                raise ValueError("Expected LINESTRING geometry")

            coords_str = geometry.upper().replace("LINESTRING", "").strip(" ()")
            fault_polygon = [
                (float(parts[0]), float(parts[1]))
                for parts in (coord.split() for coord in coords_str.split(","))
                if len(parts) >= 2
            ]

            if len(fault_polygon) < 3:
                raise ValueError(
                    f"Polygon requires at least 3 points, got {len(fault_polygon)}"
                )

            for lon, lat in fault_polygon:
                if not (-180 <= lon <= 180 and -90 <= lat <= 90):
                    raise ValueError(f"Invalid coordinate ({lon}, {lat})")

            sources_polygon.append(fault_polygon)
        except (ValueError, TypeError, IndexError) as exc:
            fault_no = fault.get("Fault Number", idx)
            raise DataValidationError(
                f"Failed to parse geometry for fault {fault_no} (row {idx}): {exc}\n"
                f"Geometry: {fault['Geometry']}"
            ) from exc

    return sources_polygon


def load_model_data(
    exposure_path: Path | str = EXPOSURE_CSV,
    faults_path: Path | str = FAULTS_XLSX,
    *,
    validate: bool = True,
) -> ModelData:
    """
    Load earthquake risk model data from CSV and Excel files.

    Args:
        exposure_path: Path to exposure inventory CSV.
        faults_path: Path to fault sources Excel file.
        validate: Run schema and quality checks before processing.

    Raises:
        FileNotFoundError: If input files do not exist.
        DataValidationError: If validation fails.
    """
    exposure_path = Path(exposure_path)
    faults_path = Path(faults_path)

    if not exposure_path.is_file():
        raise FileNotFoundError(f"Exposure file not found: {exposure_path}")
    if not faults_path.is_file():
        raise FileNotFoundError(f"Faults file not found: {faults_path}")

    logger.info("Loading exposure data from %s", exposure_path)
    exposure_df = pd.read_csv(exposure_path)
    if validate:
        validate_exposure_data(exposure_df)

    provinces = exposure_df["Province"].values
    sites_np = exposure_df[SITE_NUMERIC_COLUMNS].values.astype(np.float32)
    sites = np.asarray(sites_np)
    site_to_province = dict(zip(exposure_df["Site_ID"], exposure_df["Province"]))

    logger.info("Loading fault data from %s", faults_path)
    faults_df = pd.read_excel(faults_path)
    if validate:
        validate_fault_data(faults_df)

    source_np = faults_df[
        ["Fault Number", "a_value", "b_value", "min_mag", "max_mag"]
    ].values.astype(np.float32)
    sources_polygon = load_polygon_source(faults_df)
    source = np.asarray(source_np)

    return ModelData(
        exposure_df=exposure_df,
        provinces=provinces,
        site_ids=sites[:, 0].astype(np.int32),
        site_long=sites[:, 1].astype(np.float32),
        site_lat=sites[:, 2].astype(np.float32),
        site_sh=sites[:, 3].astype(np.float32),
        site_sm=sites[:, 4].astype(np.float32),
        site_sl=sites[:, 5].astype(np.float32),
        site_ch=sites[:, 6].astype(np.float32),
        site_cm=sites[:, 7].astype(np.float32),
        site_cl=sites[:, 8].astype(np.float32),
        site_mm=sites[:, 9].astype(np.float32),
        site_ml=sites[:, 10].astype(np.float32),
        site_sp=sites[:, 11].astype(np.float32),
        site_cp=sites[:, 12].astype(np.float32),
        site_mp=sites[:, 13].astype(np.float32),
        site_to_province=site_to_province,
        source_ids=source[:, 0].astype(np.int32),
        source_a=source[:, 1].astype(np.float32),
        source_b=source[:, 2].astype(np.float32),
        source_m_min=source[:, 3].astype(np.float32),
        source_m_max=source[:, 4].astype(np.float32),
        sources_polygon=sources_polygon,
    )
