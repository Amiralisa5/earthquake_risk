"""Single-iteration hazard, vulnerability, and loss simulation."""

import numpy as np

from config import MAX_DISTANCE_KM
from risk.distance_calculator import calculate_distance
from risk.earthquake_generator import (
    generate_location,
    generate_magnitude,
    generate_number_eq,
)
from risk.pga_calculator import calculate_pga
from risk.risk_calculator import loss_run, vulnerability_run


def hazard_run(
    source_ids,
    source_a,
    source_b,
    source_m_min,
    source_m_max,
    sources_polygon,
    gmpe_data,
    site_ids,
    site_long,
    site_lat,
    max_distance_km: float = MAX_DISTANCE_KM,
):
    number_earthquake = generate_number_eq(source_a, source_b, source_m_min)
    number_eq_per_iter = np.sum(number_earthquake)

    results_mags = []
    long_eq = []
    lat_eq = []

    for idx in range(source_ids.size):
        k = int(number_earthquake[idx].item())
        if k > 0:
            source_id = int(source_ids[idx].item())
            b = float(source_b[idx].item())
            mmin = float(source_m_min[idx].item())
            mmax = float(source_m_max[idx].item())
            mags = generate_magnitude(k, b, mmin, mmax)
            for mag in mags:
                results_mags.append(mag)
                lon, lat = generate_location(sources_polygon[source_id - 1])
                long_eq.append(lon)
                lat_eq.append(lat)

    results_mags = np.asarray(results_mags, dtype=np.float32)
    long_eq = np.asarray(long_eq, dtype=np.float32)
    lat_eq = np.asarray(lat_eq, dtype=np.float32)

    lon_src = long_eq[:, None]
    lat_src = lat_eq[:, None]
    lon_site = site_long[None, :]
    lat_site = site_lat[None, :]
    dist_mat = calculate_distance(lon_src, lat_src, lon_site, lat_site)

    mask = dist_mat <= max_distance_km
    idx_eq, idx_site = np.where(mask)

    distances = dist_mat[idx_eq, idx_site]
    magnitudes = results_mags[idx_eq]
    pga = calculate_pga(magnitudes, distances, gmpe_data)

    number_sites = len(site_ids)
    pga_max_all = np.zeros(number_sites, dtype=pga.dtype)
    np.maximum.at(pga_max_all, idx_site, pga)

    return pga_max_all, number_eq_per_iter


def hazard_process_iteration(
    source_ids,
    source_a,
    source_b,
    source_m_min,
    source_m_max,
    sources_polygon,
    gmpe_data,
    site_ids,
    site_long,
    site_lat,
    iteration,
):
    return hazard_run(
        source_ids,
        source_a,
        source_b,
        source_m_min,
        source_m_max,
        sources_polygon,
        gmpe_data,
        site_ids,
        site_long,
        site_lat,
    )


def vulnerability_process_iteration(
    source_ids,
    source_a,
    source_b,
    source_m_min,
    source_m_max,
    sources_polygon,
    gmpe_data,
    site_ids,
    site_long,
    site_lat,
    building_type_list,
    vul_data,
    vul_thresholds,
    iteration,
):
    pga_max, number_eq_per_iter = hazard_run(
        source_ids,
        source_a,
        source_b,
        source_m_min,
        source_m_max,
        sources_polygon,
        gmpe_data,
        site_ids,
        site_long,
        site_lat,
    )
    vulnerability_per_grid = vulnerability_run(
        pga_max, building_type_list, vul_data, vul_thresholds
    )
    return pga_max, number_eq_per_iter, vulnerability_per_grid


def loss_process_iteration(
    source_ids,
    source_a,
    source_b,
    source_m_min,
    source_m_max,
    sources_polygon,
    gmpe_data,
    site_ids,
    site_long,
    site_lat,
    building_type_list,
    vul_data,
    vul_thresholds,
    num_units,
    price_units,
    area_units,
    iteration,
):
    pga_max, number_eq_per_iter = hazard_run(
        source_ids,
        source_a,
        source_b,
        source_m_min,
        source_m_max,
        sources_polygon,
        gmpe_data,
        site_ids,
        site_long,
        site_lat,
    )
    vul, loss, loss_per_bt, total_loss = loss_run(
        pga_max,
        building_type_list,
        num_units,
        price_units,
        area_units,
        vul_data,
        vul_thresholds,
    )
    return pga_max, number_eq_per_iter, vul, loss, loss_per_bt, total_loss
