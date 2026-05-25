"""Run the full earthquake risk pipeline with logging and optimized structure."""

import logging
from typing import NamedTuple

import numpy as np
import pandas as pd

from config import MAX_DISTANCE_KM
from risk.data_loader import load_model_data
from risk.distance_calculator import calculate_distance
from risk.earthquake_generator import generate_earthquakes
from risk.pga_calculator import calculate_pga, load_gmpe_data
from risk.risk_calculator import calculate_risk, calculate_vul, load_vulnerability_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Building types and their exposure mapping
BUILDING_TYPES = ["SH", "SM", "SL", "CH", "CM", "CL", "MM", "ML"]
BUILDING_TYPE_MAPPING = {
    "SH": ("site_sh", "site_sp"),  # (count attribute, price attribute)
    "SM": ("site_sm", "site_sp"),
    "SL": ("site_sl", "site_sp"),
    "CH": ("site_ch", "site_cp"),
    "CM": ("site_cm", "site_cp"),
    "CL": ("site_cl", "site_cp"),
    "MM": ("site_mm", "site_mp"),
    "ML": ("site_ml", "site_mp"),
}


class PipelineResults(NamedTuple):
    """Container for pipeline output."""
    total_loss: float
    loss_per_site: np.ndarray
    loss_by_province: list
    earthquake_catalog_size: int


def validate_inputs():
    """Validate that all required input files exist."""
    from config import EXPOSURE_CSV, FAULTS_XLSX, GMPE_MAT, VULNERABILITY_XLSX

    required_files = {
        "Exposure data": EXPOSURE_CSV,
        "Fault sources": FAULTS_XLSX,
        "GMPE lookup table": GMPE_MAT,
        "Vulnerability curves": VULNERABILITY_XLSX,
    }

    for name, filepath in required_files.items():
        if not filepath.exists():
            logger.error(f"Missing required file: {name} at {filepath}")
            raise FileNotFoundError(f"Required file not found: {filepath}")
        logger.info(f"✓ Found {name}: {filepath}")


def load_pipeline_data():
    """Load all static data (model, GMPE, vulnerability)."""
    logger.info("Loading model data...")
    model = load_model_data()
    logger.info(
        f"  Loaded {len(model.site_ids)} exposure sites from {len(model.provinces)} provinces"
    )
    logger.info(f"  Loaded {len(model.source_ids)} fault sources")

    logger.info("Loading GMPE data...")
    gmpe = load_gmpe_data()
    logger.info(f"  GMPE table shape: {gmpe['Acc'].shape}")

    logger.info("Loading vulnerability data...")
    vul_data = load_vulnerability_data()
    logger.info(f"  Vulnerability table loaded")

    return model, gmpe, vul_data


def generate_catalog(model):
    """Generate stochastic earthquake catalog."""
    logger.info("Generating earthquake catalog...")
    results_mags, results_sources, long_eq, lat_eq = generate_earthquakes(model)

    logger.info(f"  Generated {len(results_mags)} earthquakes")
    logger.info(f"  Magnitude range: [{results_mags.min():.2f}, {results_mags.max():.2f}]")
    logger.info(f"  Longitude range: [{long_eq.min():.2f}, {long_eq.max():.2f}]")
    logger.info(f"  Latitude range: [{lat_eq.min():.2f}, {lat_eq.max():.2f}]")

    return results_mags, results_sources, long_eq, lat_eq


def compute_distance_matrix(model, long_eq, lat_eq):
    """Compute distance matrix between earthquakes and sites."""
    logger.info("Computing distance matrix...")

    # Reshape for broadcasting: (n_earthquakes, 1) x (1, n_sites)
    lon_src = long_eq[:, None]
    lat_src = lat_eq[:, None]
    lon_site = model.site_long[None, :]
    lat_site = model.site_lat[None, :]

    dist_mat = calculate_distance(lon_src, lat_src, lon_site, lat_site)
    logger.info(f"  Distance matrix shape: {dist_mat.shape}")

    # Find nearby earthquake-site pairs
    mask = dist_mat <= MAX_DISTANCE_KM
    idx_eq, idx_site = np.where(mask)

    logger.info(
        f"  Found {len(idx_eq)} earthquake-site pairs within {MAX_DISTANCE_KM} km"
    )

    return dist_mat, idx_eq, idx_site


def compute_ground_motion(results_mags, dist_mat, idx_eq, idx_site, gmpe):
    """Compute PGA for nearby earthquake-site pairs."""
    logger.info("Computing ground motion (PGA)...")

    distances = dist_mat[idx_eq, idx_site]
    magnitudes = results_mags[idx_eq]
    pga = calculate_pga(magnitudes, distances, gmpe)

    logger.info(f"  PGA range: [{pga.min():.4f}, {pga.max():.4f}] (g)")

    return pga


def aggregate_max_pga(pga, idx_site, num_sites):
    """Aggregate maximum PGA across all earthquakes for each site."""
    logger.info("Aggregating maximum PGA per site...")

    pga_max_all = np.zeros(num_sites, dtype=pga.dtype)
    np.maximum.at(pga_max_all, idx_site, pga)

    sites_affected = np.count_nonzero(pga_max_all)
    logger.info(f"  {sites_affected}/{num_sites} sites affected by earthquakes")
    logger.info(f"  Max PGA across all sites: {pga_max_all.max():.4f} (g)")

    return pga_max_all


def compute_vulnerabilities(pga_max_all, vul_data):
    """Compute vulnerability for all building types."""
    logger.info("Computing vulnerability curves...")

    vulnerabilities = {}
    for building_type in BUILDING_TYPES:
        vulnerabilities[building_type] = calculate_vul(
            pga_max_all, building_type, vul_data
        )

    logger.info(f"  Computed vulnerabilities for {len(vulnerabilities)} building types")

    return vulnerabilities


def compute_losses(model, vulnerabilities):
    """Compute monetary losses for all building types."""
    logger.info("Computing losses...")

    losses = {}
    total_loss_per_building = {}

    for building_type in BUILDING_TYPES:
        count_attr, price_attr = BUILDING_TYPE_MAPPING[building_type]
        count = getattr(model, count_attr)
        price = getattr(model, price_attr)

        losses[building_type] = calculate_risk(vulnerabilities[building_type], count, price)
        total_loss_per_building[building_type] = losses[building_type].sum()

    logger.info("  Loss by building type:")
    for building_type, total in sorted(
        total_loss_per_building.items(), key=lambda x: x[1], reverse=True
    ):
        logger.info(f"    {building_type}: ${total / 1e9:.3f}B")

    return losses


def aggregate_losses(model, losses):
    """Aggregate total loss and loss by province."""
    logger.info("Aggregating losses...")

    # Total loss across all building types and sites
    loss_all_per_grid = np.sum(
        [losses[bt] for bt in BUILDING_TYPES], axis=0
    )
    loss_all = np.sum(loss_all_per_grid)

    logger.info(f"  Total loss: ${loss_all / 1e9:.3f}B")

    # Loss by province
    results_df = pd.DataFrame(
        {
            "Site_ID": model.site_ids,
            "Loss": loss_all_per_grid,
        }
    )
    results_df["Province"] = (
        results_df["Site_ID"].map(model.site_to_province).fillna("Unknown")
    )
    loss_by_province = results_df.groupby("Province")["Loss"].sum().reset_index()
    loss_by_province = loss_by_province.sort_values("Loss", ascending=False)

    logger.info(f"  Loss by province (top 5):")
    for _, row in loss_by_province.head(5).iterrows():
        logger.info(f"    {row['Province']}: ${row['Loss'] / 1e9:.3f}B")

    return loss_all_per_grid, loss_all, loss_by_province.values.tolist()


def run():
    """Execute the full earthquake risk pipeline."""
    logger.info("=" * 80)
    logger.info("STARTING EARTHQUAKE RISK MODEL PIPELINE")
    logger.info("=" * 80)

    try:
        # Step 1: Validate inputs
        validate_inputs()

        # Step 2: Load data
        model, gmpe, vul_data = load_pipeline_data()

        # Step 3: Generate earthquake catalog
        results_mags, results_sources, long_eq, lat_eq = generate_catalog(model)
        earthquake_count = len(results_mags)

        # Step 4: Distance calculations
        dist_mat, idx_eq, idx_site = compute_distance_matrix(model, long_eq, lat_eq)

        # Step 5: Ground motion computation
        pga = compute_ground_motion(results_mags, dist_mat, idx_eq, idx_site, gmpe)

        # Step 6: Aggregate PGA
        pga_max_all = aggregate_max_pga(pga, idx_site, len(model.site_ids))

        # Step 7: Vulnerability curves
        vulnerabilities = compute_vulnerabilities(pga_max_all, vul_data)

        # Step 8: Loss calculations
        losses = compute_losses(model, vulnerabilities)

        # Step 9: Aggregate results
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

    except FileNotFoundError as e:
        logger.error(f"File error: {e}")
        raise
    except ValueError as e:
        logger.error(f"Computation error: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    results = run()
    print("\n📊 FINAL RESULTS:")
    print(f"  Total Loss: ${results.total_loss / 1e9:.3f}B")
    print(f"  Earthquakes Generated: {results.earthquake_catalog_size}")
    print(f"  Sites Analyzed: {len(results.loss_per_site)}")
