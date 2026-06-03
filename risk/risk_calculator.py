"""Vulnerability curves, thresholds, and monetary loss per site."""

import numpy as np
import pandas as pd

from config import VULNERABILITY_XLSX, VUL_THRESHOLD_CSV

_vul_cache: dict | None = None
_threshold_cache: dict | None = None


def load_vulnerability_data(vul_path=VULNERABILITY_XLSX) -> dict:
    global _vul_cache
    if _vul_cache is None:
        df = pd.read_excel(vul_path)
        _vul_cache = {
            "df": df,
            "pga_vals": np.asarray(df["PGA"].values),
        }
    return _vul_cache


def load_vul_thresholds(threshold_path=VUL_THRESHOLD_CSV) -> dict:
    global _threshold_cache
    if _threshold_cache is None:
        df = pd.read_csv(threshold_path)
        _threshold_cache = df.iloc[0].to_dict()
    return _threshold_cache


def calculate_vul(pga_value, building_type, vul_data=None, thresholds=None):
    if vul_data is None:
        vul_data = load_vulnerability_data()
    if thresholds is None:
        thresholds = load_vul_thresholds()

    df = vul_data["df"]
    pga_vals = vul_data["pga_vals"]
    vul = np.interp(pga_value, pga_vals, df[building_type].values)

    threshold = thresholds[building_type]
    return np.where(vul >= threshold, np.float32(1.0), vul)


def calculate_risk(vul, number, price, area):
    """Loss = vulnerability × unit count × unit price × unit area."""
    return vul * area * number * price


def vulnerability_run(pga_value, building_type_list, vul_data=None, thresholds=None):
    return {
        bt: calculate_vul(pga_value, bt, vul_data, thresholds)
        for bt in building_type_list
    }


def loss_run(
    pga_value,
    building_type_list,
    num_units,
    price_units,
    area_units,
    vul_data=None,
    thresholds=None,
):
    vul = {}
    loss = {}
    loss_per_bt = {}
    total_loss = np.float32(0.0)

    for bt in building_type_list:
        vul[bt] = calculate_vul(pga_value, bt, vul_data, thresholds)
        loss[bt] = calculate_risk(
            vul[bt], num_units[bt], price_units[bt], area_units[bt]
        )
        loss_per_bt[bt] = np.sum(loss[bt])
        total_loss += loss_per_bt[bt]

    return vul, loss, loss_per_bt, total_loss
