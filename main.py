"""Run the full earthquake risk pipeline."""

import logging

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

from config import MAX_DISTANCE_KM
from risk.data_loader import load_model_data
from risk.distance_calculator import calculate_distance
from risk.earthquake_generator import generate_earthquakes
from risk.pga_calculator import calculate_pga, load_gmpe_data
from risk.risk_calculator import calculate_risk, calculate_vul, load_vulnerability_data


def run():
    model = load_model_data()
    gmpe = load_gmpe_data()
    vul_data = load_vulnerability_data()

    results_mags, results_sources, long_eq, lat_eq = generate_earthquakes(model)

    print(results_mags)
    print(results_sources)
    print(long_eq)
    print(lat_eq)
    print(len(results_mags))
    print(len(results_sources))
    print(len(long_eq))
    print(len(lat_eq))
    print("----------")

    print(long_eq.shape)
    print(lat_eq.shape)
    print(model.site_long.shape)
    print(model.site_lat.shape)

    lon_src = long_eq[:, None]
    lat_src = lat_eq[:, None]
    lon_site = model.site_long[None, :]
    lat_site = model.site_lat[None, :]
    dist_mat = calculate_distance(lon_src, lat_src, lon_site, lat_site)
    print("dist_mat shape:", dist_mat.shape)

    mask = dist_mat <= MAX_DISTANCE_KM
    idx_eq, idx_site = np.where(mask)

    distances = dist_mat[idx_eq, idx_site]
    magnitudes = results_mags[idx_eq]
    pga = calculate_pga(magnitudes, distances, gmpe)

    print(distances.shape)
    print("##############")
    print(idx_site.shape)
    print(idx_site)
    print("##############")
    print(magnitudes.shape)
    print(pga)

    number_sites = len(model.site_ids)
    print(number_sites)

    pga_max_all = np.zeros(number_sites, dtype=pga.dtype)
    np.maximum.at(pga_max_all, idx_site, pga)

    vul_sh = calculate_vul(pga_max_all, "SH", vul_data)
    vul_sm = calculate_vul(pga_max_all, "SM", vul_data)
    vul_sl = calculate_vul(pga_max_all, "SL", vul_data)
    vul_ch = calculate_vul(pga_max_all, "CH", vul_data)
    vul_cm = calculate_vul(pga_max_all, "CM", vul_data)
    vul_cl = calculate_vul(pga_max_all, "CL", vul_data)
    vul_mm = calculate_vul(pga_max_all, "MM", vul_data)
    vul_ml = calculate_vul(pga_max_all, "ML", vul_data)

    loss_sh = calculate_risk(vul_sh, model.site_sh, model.site_sp)
    loss_sm = calculate_risk(vul_sm, model.site_sm, model.site_sp)
    loss_sl = calculate_risk(vul_sl, model.site_sl, model.site_sp)
    loss_ch = calculate_risk(vul_ch, model.site_ch, model.site_cp)
    loss_cm = calculate_risk(vul_cm, model.site_cm, model.site_cp)
    loss_cl = calculate_risk(vul_cl, model.site_cl, model.site_cp)
    loss_mm = calculate_risk(vul_mm, model.site_mm, model.site_mp)
    loss_ml = calculate_risk(vul_ml, model.site_ml, model.site_mp)

    loss_all_per_grid = (
        loss_sh + loss_sm + loss_sl + loss_ch + loss_cm + loss_cl + loss_mm + loss_ml
    )
    print(loss_all_per_grid.shape)

    loss_all = np.sum(loss_all_per_grid)
    print(loss_all / 1e9)

    results_df = pd.DataFrame(
        {
            "Site_ID": model.site_ids,
            "Loss": loss_all_per_grid,
        }
    )
    results_df["Province"] = (
        results_df["Site_ID"].map(model.site_to_province).fillna("Unknown")
    )
    loss_by_province = results_df.groupby("Province")["Loss"].sum()
    loss_province = loss_by_province.reset_index().values.tolist()
    print("Loss by province (sample):", loss_province[:5])

    return {
        "total_loss": loss_all,
        "loss_per_site": loss_all_per_grid,
        "loss_by_province": loss_province,
    }


if __name__ == "__main__":
    run()
