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

# Vulnerability curve damage caps (calculate_vul thresholds)
VUL_DAMAGE_THRESHOLD_STRUCTURAL = 0.5
VUL_DAMAGE_THRESHOLD_MASONRY = 0.25

STRUCTURAL_BUILDING_TYPES: frozenset[str] = frozenset(
    {"SH", "SM", "SL", "CH", "CM", "CL"}
)
MASONRY_BUILDING_TYPES: frozenset[str] = frozenset({"MM", "ML"})

VULNERABILITY_BUILDING_TYPES: tuple[str, ...] = tuple(
    sorted(STRUCTURAL_BUILDING_TYPES | MASONRY_BUILDING_TYPES)
)

# Per building-type damage cap thresholds (used by risk_calculator)
VULNERABILITY_THRESHOLDS: dict[str, float] = {
    **{bt: VUL_DAMAGE_THRESHOLD_STRUCTURAL for bt in STRUCTURAL_BUILDING_TYPES},
    **{bt: VUL_DAMAGE_THRESHOLD_MASONRY for bt in MASONRY_BUILDING_TYPES},
}
