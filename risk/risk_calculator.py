"""Vulnerability curves and monetary loss per site."""

import numpy as np
import pandas as pd

from config import VULNERABILITY_XLSX

_vul_cache: dict | None = None


def load_vulnerability_data(vul_path=VULNERABILITY_XLSX) -> dict:
    global _vul_cache
    if _vul_cache is None:
        df = pd.read_excel(vul_path)
        _vul_cache = {
            "df": df,
            "pga_vals": np.asarray(df["PGA"].values),
        }
    return _vul_cache


def calculate_vul(pga_value, building_type, vul_data=None):
    if vul_data is None:
        vul_data = load_vulnerability_data()

    df = vul_data["df"]
    pga_vals = vul_data["pga_vals"]
    vul_vals = np.asarray(df[building_type].values)

    id_x = np.searchsorted(pga_vals, pga_value) - 1
    id_x = np.clip(id_x, 0, len(pga_vals) - 2)

    x0 = pga_vals[id_x]
    x1 = pga_vals[id_x + 1]
    y0 = vul_vals[id_x]
    y1 = vul_vals[id_x + 1]

    slope = (y1 - y0) / (x1 - x0 + np.float32(1e-6))
    vul = y0 + slope * (pga_value - x0)

    thresholded = 0.5 if building_type in ["SH", "SM", "SL", "CH", "CM", "CL"] else 0.25
    return np.where(vul >= thresholded, np.float32(1.0), vul)


def calculate_risk(vul, number, price):
    """Loss = vulnerability × unit count × unit price."""
    return vul * number * price
