"""Vulnerability curves, thresholds, and monetary loss per site."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from config import VULNERABILITY_XLSX, VUL_THRESHOLD_CSV


@lru_cache(maxsize=4)
def load_vulnerability_data(vul_path: Path = VULNERABILITY_XLSX) -> dict:
    df = pd.read_excel(vul_path)
    return {
        "df": df,
        "pga_vals": np.asarray(df["PGA"].values),
    }


@lru_cache(maxsize=4)
def load_vul_thresholds(threshold_path: Path = VUL_THRESHOLD_CSV) -> dict:
    df = pd.read_csv(threshold_path)
    return df.iloc[0].to_dict()


def calculate_vul(
    pga_value: np.ndarray,
    building_type: str,
    vul_data: dict | None = None,
    thresholds: dict | None = None,
) -> np.ndarray:
    if vul_data is None:
        vul_data = load_vulnerability_data()
    if thresholds is None:
        thresholds = load_vul_thresholds()

    df = vul_data["df"]
    pga_vals = vul_data["pga_vals"]
    vul = np.interp(pga_value, pga_vals, df[building_type].values)

    threshold = thresholds[building_type]
    return np.where(vul >= threshold, np.float32(1.0), vul)


def calculate_risk(
    vul: np.ndarray,
    number: np.ndarray,
    price: np.ndarray,
    area: np.ndarray,
) -> np.ndarray:
    """Loss = vulnerability × unit count × unit price × unit area."""
    return vul * area * number * price


def vulnerability_run(
    pga_value: np.ndarray,
    building_type_list: list[str],
    vul_data: dict | None = None,
    thresholds: dict | None = None,
) -> dict[str, np.ndarray]:
    return {
        bt: calculate_vul(pga_value, bt, vul_data, thresholds)
        for bt in building_type_list
    }


def loss_run(
    pga_value: np.ndarray,
    building_type_list: list[str],
    num_units: dict[str, np.ndarray],
    price_units: dict[str, np.ndarray],
    area_units: dict[str, np.ndarray],
    vul_data: dict | None = None,
    thresholds: dict | None = None,
) -> tuple[dict, dict, dict, np.floating]:
    vul: dict[str, np.ndarray] = {}
    loss: dict[str, np.ndarray] = {}
    loss_per_bt: dict[str, np.floating] = {}
    total_loss = np.float32(0.0)

    for bt in building_type_list:
        vul[bt] = calculate_vul(pga_value, bt, vul_data, thresholds)
        loss[bt] = calculate_risk(
            vul[bt], num_units[bt], price_units[bt], area_units[bt]
        )
        loss_per_bt[bt] = np.sum(loss[bt])
        total_loss += loss_per_bt[bt]

    return vul, loss, loss_per_bt, total_loss
