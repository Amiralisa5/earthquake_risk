"""Legacy module name — re-exports from risk.pga_calculator."""

from risk.pga_calculator import calculate_pga, load_gmpe_data

__all__ = ["calculate_pga", "load_gmpe_data"]
