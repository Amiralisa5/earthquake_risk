"""Legacy module name — re-exports from risk.earthquake_generator."""

from risk.earthquake_generator import (
    PolygonPointSampler,
    calculate_magnitude_distribution,
    generate_earthquakes,
    generate_location,
    generate_magnitude,
    generate_number_eq,
)

__all__ = [
    "PolygonPointSampler",
    "calculate_magnitude_distribution",
    "generate_number_eq",
    "generate_magnitude",
    "generate_location",
    "generate_earthquakes",
]
