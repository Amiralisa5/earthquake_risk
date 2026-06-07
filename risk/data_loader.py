"""Load exposure sites, fault sources, and derived numpy arrays."""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from shapely.geometry import Point, Polygon

from config import (
    AREA_EXPOSURE_CSV,
    BUILDING_EXPOSURE_CSV,
    BUILDING_TYPES,
    COST_EXPOSURE_CSV,
    FAULTS_XLSX,
    SITE_CSV,
)


@dataclass
class ModelData:
    site_df: pd.DataFrame
    provinces: np.ndarray
    site_ids: np.ndarray
    site_long: np.ndarray
    site_lat: np.ndarray
    site_to_province: dict
    building_types: list
    building_counts: dict
    building_costs: dict
    building_areas: dict
    source_ids: np.ndarray
    source_a: np.ndarray
    source_b: np.ndarray
    source_m_min: np.ndarray
    source_m_max: np.ndarray
    sources_polygon: list


def load_polygon_source(faults_df: pd.DataFrame) -> list:
    sources_polygon = []
    for row_idx, fault in faults_df.iterrows():
        geometry = fault["Geometry"]
        try:
            fault_polygon = [
                (float(coord.split()[0]), float(coord.split()[1]))
                for coord in geometry.strip("LINESTRING ()").split(",")
            ]
        except (ValueError, IndexError) as exc:
            raise ValueError(
                f"Malformed geometry in fault row {row_idx} "
                f"(Fault Number={fault.get('Fault Number', '?')}): {geometry!r}"
            ) from exc
        if len(fault_polygon) < 3:
            raise ValueError(
                f"Fault row {row_idx} has fewer than 3 coordinate pairs — "
                "cannot form a valid polygon."
            )
        sources_polygon.append(fault_polygon)
    return sources_polygon


def _load_exposure_tables(
    site_path=SITE_CSV,
    building_path=BUILDING_EXPOSURE_CSV,
    cost_path=COST_EXPOSURE_CSV,
    area_path=AREA_EXPOSURE_CSV,
):
    site_df = pd.read_csv(site_path).sort_values("Site_ID").reset_index(drop=True)
    building_df = pd.read_csv(building_path).sort_values("Site_ID").reset_index(drop=True)
    cost_df = pd.read_csv(cost_path).sort_values("Site_ID").reset_index(drop=True)
    area_df = pd.read_csv(area_path).sort_values("Site_ID").reset_index(drop=True)

    site_id_set = set(site_df["Site_ID"])
    for label, df in [
        ("building_exposure", building_df),
        ("cost_exposure", cost_df),
        ("area_exposure", area_df),
    ]:
        df_id_set = set(df["Site_ID"])
        extra_in_site = site_id_set - df_id_set
        extra_in_df = df_id_set - site_id_set
        if extra_in_site or extra_in_df:
            raise ValueError(
                f"Site_ID mismatch between site.csv and {label}.csv — "
                f"missing from {label}: {extra_in_site}, "
                f"extra in {label}: {extra_in_df}"
            )

    building_types = [c for c in building_df.columns if c != "Site_ID"]
    if building_types != BUILDING_TYPES:
        raise ValueError(
            f"Unexpected building types in exposure data: {building_types}"
        )

    building_counts = {
        bt: building_df[bt].to_numpy(dtype=np.int32) for bt in building_types
    }
    building_costs = {
        bt: cost_df[bt].to_numpy(dtype=np.float32) for bt in building_types
    }
    building_areas = {
        bt: area_df[bt].to_numpy(dtype=np.float32) for bt in building_types
    }

    return site_df, building_types, building_counts, building_costs, building_areas


def load_model_data(
    site_path=SITE_CSV,
    building_path=BUILDING_EXPOSURE_CSV,
    cost_path=COST_EXPOSURE_CSV,
    area_path=AREA_EXPOSURE_CSV,
    faults_path=FAULTS_XLSX,
) -> ModelData:
    site_df, building_types, building_counts, building_costs, building_areas = (
        _load_exposure_tables(site_path, building_path, cost_path, area_path)
    )

    provinces = site_df["Province"].values
    site_to_province = dict(zip(site_df["Site_ID"], site_df["Province"]))

    faults_df = pd.read_excel(faults_path)
    source_np = faults_df[
        ["Fault Number", "a_value", "b_value", "min_mag", "max_mag"]
    ].values.astype(np.float32)

    sources_polygon = load_polygon_source(faults_df)
    source = np.asarray(source_np)

    return ModelData(
        site_df=site_df,
        provinces=provinces,
        site_ids=site_df["Site_ID"].to_numpy(dtype=np.int32),
        site_long=site_df["Longitude"].to_numpy(dtype=np.float32),
        site_lat=site_df["Latitude"].to_numpy(dtype=np.float32),
        site_to_province=site_to_province,
        building_types=building_types,
        building_counts=building_counts,
        building_costs=building_costs,
        building_areas=building_areas,
        source_ids=source[:, 0].astype(np.int32),
        source_a=source[:, 1].astype(np.float32),
        source_b=source[:, 2].astype(np.float32),
        source_m_min=source[:, 3].astype(np.float32),
        source_m_max=source[:, 4].astype(np.float32),
        sources_polygon=sources_polygon,
    )
