"""Run a single-iteration earthquake risk pipeline with logging."""
import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import (
    AREA_EXPOSURE_CSV,
    BUILDING_EXPOSURE_CSV,
    BUILDING_TYPES,
    COST_EXPOSURE_CSV,
    FAULTS_XLSX,
    GMPE_MAT,
    MAX_DISTANCE_KM,
    SITE_CSV,
    VULNERABILITY_XLSX,
    VUL_THRESHOLD_CSV,
)
from risk.data_loader import load_model_data
from risk.distance_calculator import calculate_distance
from risk.earthquake_generator import generate_earthquakes
from risk.pga_calculator import calculate_pga, load_gmpe_data
from risk.risk_calculator import (
    calculate_risk,
    calculate_vul,
    load_vul_thresholds,
    load_vulnerability_data,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


class PipelineResults(NamedTuple):
    total_loss: float
    loss_per_site: np.ndarray
    loss_by_province: list
    earthquake_catalog_size: int


def validate_inputs():
    required_files = {
        "Site locations": SITE_CSV,
        "Building exposure": BUILDING_EXPOSURE_CSV,
        "Cost exposure": COST_EXPOSURE_CSV,
        "Area exposure": AREA_EXPOSURE_CSV,
        "Vulnerability thresholds": VUL_THRESHOLD_CSV,
        "Fault sources": FAULTS_XLSX,
        "GMPE lookup table": GMPE_MAT,
        "Vulnerability curves": VULNERABILITY_XLSX,
    }

    for name, filepath in required_files.items():
        if not filepath.exists():
            logger.error("Missing required file: %s at %s", name, filepath)
            raise FileNotFoundError(f"Required file not found: {filepath}")
        logger.info("Found %s: %s", name, filepath)


def load_pipeline_data():
    logger.info("Loading model data...")
    model = load_model_data()
    logger.info(
        "  Loaded %s exposure sites from %s provinces",
        len(model.site_ids),
        len(set(model.provinces)),
    )
    logger.info("  Loaded %s fault sources", len(model.source_ids))

    logger.info("Loading GMPE data...")
    gmpe = load_gmpe_data()
    logger.info("  GMPE table shape: %s", gmpe["Acc"].shape)

    logger.info("Loading vulnerability data...")
    vul_data = load_vulnerability_data()
    thresholds = load_vul_thresholds()
    logger.info("  Vulnerability thresholds: %s", thresholds)

    return model, gmpe, vul_data, thresholds


def generate_catalog(model):
    logger.info("Generating earthquake catalog...")
    results_mags, results_sources, long_eq, lat_eq = generate_earthquakes(model)

    logger.info("  Generated %s earthquakes", len(results_mags))
    if len(results_mags) > 0:
        logger.info(
            "  Magnitude range: [%.2f, %.2f]",
            results_mags.min(),
            results_mags.max(),
        )

    return results_mags, results_sources, long_eq, lat_eq


def compute_distance_matrix(model, long_eq, lat_eq):
    logger.info("Computing distance matrix...")

    lon_src = long_eq[:, None]
    lat_src = lat_eq[:, None]
    lon_site = model.site_long[None, :]
    lat_site = model.site_lat[None, :]

    dist_mat = calculate_distance(lon_src, lat_src, lon_site, lat_site)
    mask = dist_mat <= MAX_DISTANCE_KM
    idx_eq, idx_site = np.where(mask)

    logger.info(
        "  Found %s earthquake-site pairs within %s km",
        len(idx_eq),
        MAX_DISTANCE_KM,
    )

    return dist_mat, idx_eq, idx_site


def compute_ground_motion(results_mags, dist_mat, idx_eq, idx_site, gmpe):
    logger.info("Computing ground motion (PGA)...")

    distances = dist_mat[idx_eq, idx_site]
    magnitudes = results_mags[idx_eq]
    pga = calculate_pga(magnitudes, distances, gmpe)

    if len(pga) > 0:
        logger.info("  PGA range: [%.4f, %.4f] (g)", pga.min(), pga.max())

    return pga


def aggregate_max_pga(pga, idx_site, num_sites):
    logger.info("Aggregating maximum PGA per site...")

    pga_max_all = np.zeros(num_sites, dtype=pga.dtype)
    if len(pga) > 0:
        np.maximum.at(pga_max_all, idx_site, pga)

    sites_affected = np.count_nonzero(pga_max_all)
    logger.info("  %s/%s sites affected by earthquakes", sites_affected, num_sites)

    return pga_max_all


def compute_vulnerabilities(pga_max_all, vul_data, thresholds):
    logger.info("Computing vulnerability curves...")

    vulnerabilities = {
        building_type: calculate_vul(
            pga_max_all, building_type, vul_data, thresholds
        )
        for building_type in BUILDING_TYPES
    }

    logger.info(
        "  Computed vulnerabilities for %s building types", len(vulnerabilities)
    )
    return vulnerabilities


def compute_losses(model, vulnerabilities):
    logger.info("Computing losses...")

    losses = {}
    total_loss_per_building = {}

    for building_type in BUILDING_TYPES:
        count = model.building_counts[building_type]
        price = model.building_costs[building_type]
        area = model.building_areas[building_type]

        losses[building_type] = calculate_risk(
            vulnerabilities[building_type], count, price, area
        )
        total_loss_per_building[building_type] = losses[building_type].sum()

    logger.info("  Loss by building type:")
    for building_type, total in sorted(
        total_loss_per_building.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info("    %s: $%.3fB", building_type, total / 1e9)

    return losses


def aggregate_losses(model, losses):
    logger.info("Aggregating losses...")

    loss_all_per_grid = np.sum([losses[bt] for bt in BUILDING_TYPES], axis=0)
    loss_all = np.sum(loss_all_per_grid)

    logger.info("  Total loss: $%.3fB", loss_all / 1e9)

    results_df = pd.DataFrame({"Site_ID": model.site_ids, "Loss": loss_all_per_grid})
    results_df["Province"] = (
        results_df["Site_ID"].map(model.site_to_province).fillna("Unknown")
    )
    loss_by_province = results_df.groupby("Province")["Loss"].sum().reset_index()
    loss_by_province = loss_by_province.sort_values("Loss", ascending=False)

    logger.info("  Loss by province (top 5):")
    for _, row in loss_by_province.head(5).iterrows():
        logger.info("    %s: $%.3fB", row["Province"], row["Loss"] / 1e9)

    return loss_all_per_grid, loss_all, loss_by_province.values.tolist()


def run():
    logger.info("=" * 80)
    logger.info("STARTING EARTHQUAKE RISK MODEL PIPELINE")
    logger.info("=" * 80)

    validate_inputs()
    model, gmpe, vul_data, thresholds = load_pipeline_data()

    results_mags, _, long_eq, lat_eq = generate_catalog(model)
    earthquake_count = len(results_mags)

    dist_mat, idx_eq, idx_site = compute_distance_matrix(model, long_eq, lat_eq)
    pga = compute_ground_motion(results_mags, dist_mat, idx_eq, idx_site, gmpe)
    pga_max_all = aggregate_max_pga(pga, idx_site, len(model.site_ids))

    vulnerabilities = compute_vulnerabilities(pga_max_all, vul_data, thresholds)
    losses = compute_losses(model, vulnerabilities)
    loss_per_site, total_loss, loss_by_province = aggregate_losses(model, losses)

    logger.info("=" * 80)
    logger.info("PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return PipelineResults(
        total_loss=total_loss,
        loss_per_site=loss_per_site,
        loss_by_province=loss_by_province,
        earthquake_catalog_size=earthquake_count,
    )


if __name__ == "__main__":
    results = run()
    print("\nFINAL RESULTS:")
    print(f"  Total Loss: ${results.total_loss / 1e9:.3f}B")
    print(f"  Earthquakes Generated: {results.earthquake_catalog_size}")
    print(f"  Sites Analyzed: {len(results.loss_per_site)}")
