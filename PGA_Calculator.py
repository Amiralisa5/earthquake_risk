"""Legacy module name — re-exports from risk.pga_calculator."""

from risk.pga_calculator import GMPEInterpolator, calculate_pga, load_gmpe_data

__all__ = ["GMPEInterpolator", "calculate_pga", "load_gmpe_data"]
