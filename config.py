"""Paths and constants for the earthquake risk model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

SITE_CSV = DATA_DIR / "site.csv"
BUILDING_EXPOSURE_CSV = DATA_DIR / "building_exposure.csv"
COST_EXPOSURE_CSV = DATA_DIR / "cost_exposure.csv"
AREA_EXPOSURE_CSV = DATA_DIR / "area_exposure.csv"
VUL_THRESHOLD_CSV = DATA_DIR / "vul_threshold.csv"

# Legacy combined exposure (optional)
EXPOSURE_CSV = DATA_DIR / "Main_Exposure.csv"

FAULTS_XLSX = DATA_DIR / "main_faults_iran_mid.xlsx"
GMPE_MAT = DATA_DIR / "GMPEs_Vs.mat"
VULNERABILITY_XLSX = DATA_DIR / "Vul1.xlsx"

BUILDING_TYPES = ["SH", "SM", "SL", "CH", "CM", "CL", "MM", "ML"]

MAX_DISTANCE_KM = 200.0

# WGS-84 semi-major axis used in the Haversine formula
EARTH_RADIUS_KM = 6378.137

# Prevents division-by-zero in bilinear PGA interpolation when table cells are identical
PGA_INTERP_EPSILON = 1e-6

RESULTS_DIR = PROJECT_ROOT / "Results"
