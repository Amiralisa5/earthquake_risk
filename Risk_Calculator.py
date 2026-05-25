"""Legacy module name — re-exports from risk.risk_calculator."""

from risk.risk_calculator import (
    VulnerabilityInterpolator,
    VulnerabilityModel,
    calculate_risk,
    calculate_vul,
    load_vulnerability_data,
)

__all__ = [
    "VulnerabilityInterpolator",
    "VulnerabilityModel",
    "calculate_risk",
    "calculate_vul",
    "load_vulnerability_data",
]
