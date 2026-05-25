"""Load exposure sites, fault sources, and derived numpy arrays."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from config import EXPOSURE_CSV, FAULTS_XLSX


@dataclass
class ModelData:
    exposure_df: pd.DataFrame
    provinces: np.ndarray
    site_ids: np.ndarray
    site_long: np.ndarray
    site_lat: np.ndarray
    site_sh: np.ndarray
    site_sm: np.ndarray
    site_sl: np.ndarray
    site_ch: np.ndarray
    site_cm: np.ndarray
    site_cl: np.ndarray
    site_mm: np.ndarray
    site_ml: np.ndarray
    site_sp: np.ndarray
    site_cp: np.ndarray
    site_mp: np.ndarray
    site_to_province: dict
    source_ids: np.ndarray
    source_a: np.ndarray
    source_b: np.ndarray
    source_m_min: np.ndarray
    source_m_max: np.ndarray
    sources_polygon: list


def load_polygon_source(faults_df: pd.DataFrame) -> list:
    sources_polygon = []
    for _, fault in faults_df.iterrows():
        geometry = fault["Geometry"]
        fault_polygon = [
            (float(coord.split()[0]), float(coord.split()[1]))
            for coord in geometry.strip("LINESTRING ()").split(",")
        ]
        sources_polygon.append(fault_polygon)
    return sources_polygon


def load_model_data(
    exposure_path=EXPOSURE_CSV,
    faults_path=FAULTS_XLSX,
) -> ModelData:
    exposure_df = pd.read_csv(exposure_path)
    provinces = exposure_df["Province"].values

    sites_np = exposure_df[
        [
            "Site_ID",
            "Longitude",
            "Latitude",
            "SH",
            "SM",
            "SL",
            "CH",
            "CM",
            "CL",
            "MM",
            "ML",
            "SP",
            "CP",
            "MP",
        ]
    ].values.astype(np.float32)

    sites = np.asarray(sites_np)
    site_to_province = dict(zip(exposure_df["Site_ID"], exposure_df["Province"]))

    faults_df = pd.read_excel(faults_path)
    source_np = faults_df[
        ["Fault Number", "a_value", "b_value", "min_mag", "max_mag"]
    ].values.astype(np.float32)

    sources_polygon = load_polygon_source(faults_df)
    source = np.asarray(source_np)

    return ModelData(
        exposure_df=exposure_df,
        provinces=provinces,
        site_ids=sites[:, 0].astype(np.int32),
        site_long=sites[:, 1].astype(np.float32),
        site_lat=sites[:, 2].astype(np.float32),
        site_sh=sites[:, 3].astype(np.float32),
        site_sm=sites[:, 4].astype(np.float32),
        site_sl=sites[:, 5].astype(np.float32),
        site_ch=sites[:, 6].astype(np.float32),
        site_cm=sites[:, 7].astype(np.float32),
        site_cl=sites[:, 8].astype(np.float32),
        site_mm=sites[:, 9].astype(np.float32),
        site_ml=sites[:, 10].astype(np.float32),
        site_sp=sites[:, 11].astype(np.float32),
        site_cp=sites[:, 12].astype(np.float32),
        site_mp=sites[:, 13].astype(np.float32),
        site_to_province=site_to_province,
        source_ids=source[:, 0].astype(np.int32),
        source_a=source[:, 1].astype(np.float32),
        source_b=source[:, 2].astype(np.float32),
        source_m_min=source[:, 3].astype(np.float32),
        source_m_max=source[:, 4].astype(np.float32),
        sources_polygon=sources_polygon,
    )
