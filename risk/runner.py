"""Monte Carlo hazard, vulnerability, and loss analysis with parallel iterations."""

import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

import numpy as np
import pandas as pd
from tqdm import tqdm

from config import BUILDING_TYPES, RESULTS_DIR
from risk.data_loader import load_model_data
from risk.pga_calculator import load_gmpe_data
from risk.risk_calculator import load_vul_thresholds, load_vulnerability_data
from risk.simulation import (
    hazard_process_iteration,
    loss_process_iteration,
    vulnerability_process_iteration,
)


def make_cols(start_iter, count):
    return [f"Iteration_{i}" for i in range(start_iter, start_iter + count)]


def run_monte_carlo(
    analysis_type="loss",
    investigation_time=10,
    block_size=10_000,
):
    """
    Run parallel Monte Carlo iterations.

    analysis_type: 'hazard', 'vul', or 'loss'
    investigation_time: number of stochastic years to simulate
    block_size: iterations per output parquet block
    """
    block_size = min(block_size, investigation_time)

    model = load_model_data()
    gmpe_data = load_gmpe_data()
    vul_data = load_vulnerability_data()
    vul_thresholds = load_vul_thresholds()

    source_ids = model.source_ids
    source_a = model.source_a
    source_b = model.source_b
    source_m_min = model.source_m_min
    source_m_max = model.source_m_max
    sources_polygon = model.sources_polygon

    site_ids = model.site_ids
    site_long = model.site_long
    site_lat = model.site_lat
    n_site = len(site_ids)
    building_types = model.building_types

    building_exposure = pd.DataFrame(
        {bt: model.building_counts[bt] for bt in building_types}
    )
    cost_exposure = pd.DataFrame(
        {bt: model.building_costs[bt] for bt in building_types}
    )
    area_exposure = pd.DataFrame(
        {bt: model.building_areas[bt] for bt in building_types}
    )

    if analysis_type == "hazard":
        _run_hazard(
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
            n_site,
            investigation_time,
            block_size,
        )
    elif analysis_type == "vul":
        _run_vulnerability(
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
            n_site,
            building_types,
            vul_data,
            vul_thresholds,
            investigation_time,
            block_size,
        )
    elif analysis_type == "loss":
        _run_loss(
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
            n_site,
            building_types,
            vul_data,
            vul_thresholds,
            building_exposure,
            cost_exposure,
            area_exposure,
            investigation_time,
            block_size,
        )
    else:
        raise ValueError(
            f"Unknown analysis_type: {analysis_type!r}. Use 'hazard', 'vul', or 'loss'."
        )


def _run_hazard(
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
    n_site,
    investigation_time,
    block_size,
):
    hazard_dir = RESULTS_DIR / "hazard"
    hazard_dir.mkdir(parents=True, exist_ok=True)

    pga_max_per_site_block = np.empty((n_site, block_size), dtype=np.float32)
    number_eq_per_iter_block = np.empty((block_size,), dtype=np.float32)

    start_iter_of_block = 1
    fill_idx = 0

    worker = partial(
        hazard_process_iteration,
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

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for iteration, result in tqdm(
            enumerate(executor.map(worker, range(investigation_time)), start=1),
            total=investigation_time,
            desc="Processing Iterations",
        ):
            local_pga_max, local_number_eq = result

            pga_max_per_site_block[:, fill_idx] = np.asarray(local_pga_max)
            number_eq_per_iter_block[fill_idx] = local_number_eq
            fill_idx += 1

            if fill_idx == block_size or iteration == investigation_time:
                _write_hazard_block(
                    hazard_dir,
                    iteration,
                    site_ids,
                    pga_max_per_site_block,
                    number_eq_per_iter_block,
                    fill_idx,
                    start_iter_of_block,
                )
                start_iter_of_block = iteration + 1
                fill_idx = 0


def _write_hazard_block(
    hazard_dir,
    iteration,
    site_ids,
    pga_block,
    number_eq_block,
    fill_idx,
    start_iter_of_block,
):
    pga_np = np.asarray(pga_block[:, :fill_idx])
    number_eq_np = np.asarray(number_eq_block[:fill_idx])

    pga_df = pd.DataFrame(pga_np, columns=make_cols(start_iter_of_block, fill_idx))
    pga_df.insert(0, "Site_ID", site_ids)
    pga_df.to_parquet(
        hazard_dir / f"pga_{iteration}.parquet",
        index=False,
        compression="snappy",
    )

    number_eq_df = pd.DataFrame(
        {
            "Iteration": range(start_iter_of_block, start_iter_of_block + fill_idx),
            "Number_EQ": number_eq_np,
        }
    )
    number_eq_df.to_parquet(
        hazard_dir / f"number_eq_{iteration}.parquet",
        index=False,
        compression="snappy",
    )


def _run_vulnerability(
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
    n_site,
    building_types,
    vul_data,
    vul_thresholds,
    investigation_time,
    block_size,
):
    hazard_dir = RESULTS_DIR / "hazard"
    vul_dir = RESULTS_DIR / "vulnerability"
    hazard_dir.mkdir(parents=True, exist_ok=True)
    vul_dir.mkdir(parents=True, exist_ok=True)

    pga_max_per_site_block = np.empty((n_site, block_size), dtype=np.float32)
    number_eq_per_iter_block = np.empty((block_size,), dtype=np.float32)
    vul_block = {bt: np.empty((n_site, block_size), dtype=np.float32) for bt in building_types}

    start_iter_of_block = 1
    fill_idx = 0

    worker = partial(
        vulnerability_process_iteration,
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
        building_types,
        vul_data,
        vul_thresholds,
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for iteration, result in tqdm(
            enumerate(executor.map(worker, range(investigation_time)), start=1),
            total=investigation_time,
            desc="Processing Iterations",
        ):
            local_pga_max, local_number_eq, local_vul = result

            pga_max_per_site_block[:, fill_idx] = np.asarray(local_pga_max)
            number_eq_per_iter_block[fill_idx] = local_number_eq
            for bt in building_types:
                vul_block[bt][:, fill_idx] = np.asarray(local_vul[bt])
            fill_idx += 1

            if fill_idx == block_size or iteration == investigation_time:
                _write_vulnerability_block(
                    hazard_dir,
                    vul_dir,
                    iteration,
                    site_ids,
                    building_types,
                    pga_max_per_site_block,
                    number_eq_per_iter_block,
                    vul_block,
                    fill_idx,
                    start_iter_of_block,
                )
                start_iter_of_block = iteration + 1
                fill_idx = 0


def _write_vulnerability_block(
    hazard_dir,
    vul_dir,
    iteration,
    site_ids,
    building_types,
    pga_block,
    number_eq_block,
    vul_block,
    fill_idx,
    start_iter_of_block,
):
    for bt in building_types:
        vul_np = np.asarray(vul_block[bt][:, :fill_idx])
        vul_df = pd.DataFrame(vul_np, columns=make_cols(start_iter_of_block, fill_idx))
        vul_df.insert(0, "Site_ID", site_ids)
        vul_df.to_parquet(
            vul_dir / f"vul_{bt}_{iteration}.parquet",
            index=False,
            compression="snappy",
        )

    pga_np = np.asarray(pga_block[:, :fill_idx])
    pga_df = pd.DataFrame(pga_np, columns=make_cols(start_iter_of_block, fill_idx))
    pga_df.insert(0, "Site_ID", site_ids)
    pga_df.to_parquet(
        hazard_dir / f"pga_{iteration}.parquet",
        index=False,
        compression="snappy",
    )

    number_eq_np = np.asarray(number_eq_block[:fill_idx])
    number_eq_df = pd.DataFrame(
        {
            "Iteration": range(start_iter_of_block, start_iter_of_block + fill_idx),
            "Number_EQ": number_eq_np,
        }
    )
    number_eq_df.to_parquet(
        hazard_dir / f"number_eq_{iteration}.parquet",
        index=False,
        compression="snappy",
    )


def _run_loss(
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
    n_site,
    building_types,
    vul_data,
    vul_thresholds,
    building_exposure,
    cost_exposure,
    area_exposure,
    investigation_time,
    block_size,
):
    hazard_dir = RESULTS_DIR / "hazard"
    vul_dir = RESULTS_DIR / "vulnerability"
    loss_dir = RESULTS_DIR / "loss"
    for d in (hazard_dir, vul_dir, loss_dir):
        d.mkdir(parents=True, exist_ok=True)

    n_building_types = len(building_types)
    pga_max_per_site_block = np.empty((n_site, block_size), dtype=np.float32)
    number_eq_per_iter_block = np.empty((block_size,), dtype=np.float32)
    loss_by_b_type_block = np.empty((n_building_types, block_size), dtype=np.float32)
    total_loss_block = np.empty((block_size,), dtype=np.float32)
    vul_block = {bt: np.empty((n_site, block_size), dtype=np.float32) for bt in building_types}
    loss_per_grid_block = {
        bt: np.empty((n_site, block_size), dtype=np.float32) for bt in building_types
    }

    start_iter_of_block = 1
    fill_idx = 0

    worker = partial(
        loss_process_iteration,
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
        building_types,
        vul_data,
        vul_thresholds,
        building_exposure,
        cost_exposure,
        area_exposure,
    )

    with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for iteration, result in tqdm(
            enumerate(executor.map(worker, range(investigation_time)), start=1),
            total=investigation_time,
            desc="Processing Iterations",
        ):
            (
                local_pga_max,
                local_number_eq,
                local_vul,
                local_loss,
                local_loss_per_bt,
                local_total_loss,
            ) = result

            pga_max_per_site_block[:, fill_idx] = np.asarray(local_pga_max)
            number_eq_per_iter_block[fill_idx] = local_number_eq
            total_loss_block[fill_idx] = local_total_loss

            loss_per_bt_array = []
            for bt in building_types:
                vul_block[bt][:, fill_idx] = np.asarray(local_vul[bt])
                loss_per_grid_block[bt][:, fill_idx] = np.asarray(local_loss[bt])
                loss_per_bt_array.append(local_loss_per_bt[bt])

            loss_by_b_type_block[:, fill_idx] = np.asarray(loss_per_bt_array)
            fill_idx += 1

            if fill_idx == block_size or iteration == investigation_time:
                _write_loss_block(
                    hazard_dir,
                    vul_dir,
                    loss_dir,
                    iteration,
                    site_ids,
                    building_types,
                    pga_max_per_site_block,
                    number_eq_per_iter_block,
                    vul_block,
                    loss_per_grid_block,
                    loss_by_b_type_block,
                    total_loss_block,
                    fill_idx,
                    start_iter_of_block,
                )
                start_iter_of_block = iteration + 1
                fill_idx = 0


def _write_loss_block(
    hazard_dir,
    vul_dir,
    loss_dir,
    iteration,
    site_ids,
    building_types,
    pga_block,
    number_eq_block,
    vul_block,
    loss_per_grid_block,
    loss_by_b_type_block,
    total_loss_block,
    fill_idx,
    start_iter_of_block,
):
    for bt in building_types:
        vul_np = np.asarray(vul_block[bt][:, :fill_idx])
        vul_df = pd.DataFrame(vul_np, columns=make_cols(start_iter_of_block, fill_idx))
        vul_df.insert(0, "Site_ID", site_ids)
        vul_df.to_parquet(
            vul_dir / f"vul_{bt}_{iteration}.parquet",
            index=False,
            compression="snappy",
        )

        loss_per_grid_np = np.asarray(loss_per_grid_block[bt][:, :fill_idx])
        loss_df = pd.DataFrame(
            loss_per_grid_np, columns=make_cols(start_iter_of_block, fill_idx)
        )
        loss_df.insert(0, "Site_ID", site_ids)
        loss_df.to_parquet(
            loss_dir / f"loss_per_grid_{bt}_{iteration}.parquet",
            index=False,
            compression="snappy",
        )

    pga_np = np.asarray(pga_block[:, :fill_idx])
    pga_df = pd.DataFrame(pga_np, columns=make_cols(start_iter_of_block, fill_idx))
    pga_df.insert(0, "Site_ID", site_ids)
    pga_df.to_parquet(
        hazard_dir / f"pga_{iteration}.parquet",
        index=False,
        compression="snappy",
    )

    loss_by_b_type_np = np.asarray(loss_by_b_type_block[:, :fill_idx])
    loss_by_b_type_df = pd.DataFrame(
        loss_by_b_type_np, columns=make_cols(start_iter_of_block, fill_idx)
    )
    loss_by_b_type_df.insert(0, "Building_Type", building_types)
    loss_by_b_type_df.to_parquet(
        loss_dir / f"loss_by_btype_{iteration}.parquet",
        index=False,
        compression="snappy",
    )

    number_eq_np = np.asarray(number_eq_block[:fill_idx])
    number_eq_df = pd.DataFrame(
        {
            "Iteration": range(start_iter_of_block, start_iter_of_block + fill_idx),
            "Number_EQ": number_eq_np,
        }
    )
    number_eq_df.to_parquet(
        hazard_dir / f"number_eq_{iteration}.parquet",
        index=False,
        compression="snappy",
    )

    total_loss_np = np.asarray(total_loss_block[:fill_idx])
    total_loss_df = pd.DataFrame(
        {
            "Iteration": range(start_iter_of_block, start_iter_of_block + fill_idx),
            "Total_Loss": total_loss_np,
        }
    )
    total_loss_df.to_parquet(
        loss_dir / f"total_loss_{iteration}.parquet",
        index=False,
        compression="snappy",
    )
