"""Ground motion (PGA) from magnitude and distance via GMPE lookup table."""

import numpy as np
from scipy.io import loadmat

from config import GMPE_MAT

_gmpe_cache: dict | None = None


def load_gmpe_data(gmpe_path=GMPE_MAT) -> dict:
    global _gmpe_cache
    if _gmpe_cache is None:
        gmpe_data = loadmat(gmpe_path)
        _gmpe_cache = {
            "Acc": gmpe_data["Acc"].astype(np.float32),
            "M_max": gmpe_data["M_max"].flatten().astype(np.float32),
            "r_jb": gmpe_data["r_jb"].flatten().astype(np.float32),
        }
    return _gmpe_cache


def calculate_pga(magnitude, distance, gmpe=None):
    if gmpe is None:
        gmpe = load_gmpe_data()

    acc = gmpe["Acc"]
    m_max = gmpe["M_max"]
    r_jb = gmpe["r_jb"]

    d_adj = distance.astype(np.float32)

    i1 = np.searchsorted(m_max, magnitude, "right")
    i0 = np.clip(i1 - 1, 0, m_max.size - 2)
    i1 = i0 + 1

    m0 = m_max[i0]
    m1 = m_max[i1]
    t = (magnitude - m0) / (m1 - m0 + 1e-6)

    j1 = np.searchsorted(r_jb, d_adj, "right")
    j0 = np.clip(j1 - 1, 0, r_jb.size - 2)
    j1 = j0 + 1

    r0 = r_jb[j0]
    r1 = r_jb[j1]
    u = (d_adj - r0) / (r1 - r0 + 1e-6)

    z00 = acc[i0, j0]
    z10 = acc[i1, j0]
    z01 = acc[i0, j1]
    z11 = acc[i1, j1]

    return (1 - t) * (1 - u) * z00 + t * (1 - u) * z10 + (1 - t) * u * z01 + t * u * z11
