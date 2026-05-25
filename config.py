"""Paths and constants for the earthquake risk model."""

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"

EXPOSURE_CSV = DATA_DIR / "Main_Exposure.csv"
FAULTS_XLSX = DATA_DIR / "main_faults_iran_mid.xlsx"
GMPE_MAT = DATA_DIR / "GMPEs_Vs.mat"
VULNERABILITY_XLSX = DATA_DIR / "Vul1.xlsx"

MAX_DISTANCE_KM = 200.0

STEEL_MODERATE_RATIO = 0.25
CONCRETE_MODERATE_RATIO = 0.25
MASONRY_MODERATE_RATIO = 0.25
