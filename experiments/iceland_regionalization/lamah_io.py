"""LAMAH-ICE loader for the Iceland jFUSE regionalization rerun.

Loads per-catchment forcing (CARRA/RAV precip, temp, PET), discharge obs
(converted to mm/day via LAMAH area_calc), and catchment attributes for the
A_basins_total_upstrm set. This is the clean, correct-by-construction data the
memory's `lamah_clean_test.py` used to reach in-sample 0.59 / kNN-LOO 0.42.

Reconstructed 2026-07-06 after the prior session's scratchpad was wiped.
"""
from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pandas as pd

LAMAH = Path("/Users/darri.eythorsson/compHydro/data/lamah_ice")
A = LAMAH / "A_basins_total_upstrm"
MET_DIR = A / "2_timeseries" / "daily" / "meteorological_data"
Q_DIR = LAMAH / "D_gauges" / "2_timeseries" / "daily"
GAUGE_ATTR = LAMAH / "D_gauges" / "1_attributes" / "Gauge_attributes.csv"
CATCH_ATTR = A / "1_attributes" / "Catchment_attributes.csv"

# Analysis window (matches the 18yr clean test that reached the 0.42 ceiling).
YEAR_START = 2000
YEAR_END = 2018  # inclusive
WARMUP_DAYS = 365  # first year excluded from scoring


@functools.lru_cache(maxsize=1)
def gauge_attrs() -> pd.DataFrame:
    df = pd.read_csv(GAUGE_ATTR, sep=";")
    return df.set_index("id")


@functools.lru_cache(maxsize=1)
def catch_attrs() -> pd.DataFrame:
    df = pd.read_csv(CATCH_ATTR, sep=";")
    return df.set_index("id")


def _date_index(df: pd.DataFrame) -> pd.DatetimeIndex:
    return pd.to_datetime(dict(year=df["YYYY"], month=df["MM"], day=df["DD"]))


def load_forcing(gid: int, precip_col: str = "prec_carra") -> pd.DataFrame:
    """Daily forcing for one gauge id over the analysis window.

    Columns: precip (mm/d), temp (C), pet (mm/d). Uses CARRA precip + CARRA
    2m temp by default (production forcing); PET from ERA5-Land FAO column.
    """
    f = pd.read_csv(MET_DIR / f"ID_{gid}.csv", sep=";")
    f.index = _date_index(f)
    temp_col = "2m_temp_carra" if "2m_temp_carra" in f.columns else "2m_temp_mean"
    pet_col = "pet" if "pet" in f.columns else "ref_et_rav"
    out = pd.DataFrame(
        {
            "precip": f[precip_col].astype(float),
            "temp": f[temp_col].astype(float),
            "pet": f[pet_col].astype(float).clip(lower=0.0),
        }
    )
    mask = (out.index.year >= YEAR_START) & (out.index.year <= YEAR_END)
    return out.loc[mask]


def load_obs_mm(gid: int) -> pd.Series:
    """Observed discharge as mm/day (qobs m3/s -> mm/d via area_calc km2)."""
    q = pd.read_csv(Q_DIR / f"ID_{gid}.csv", sep=";")
    q.index = _date_index(q)
    qobs = q["qobs"].astype(float)
    qobs[qobs < 0] = np.nan  # LAMAH uses negative sentinels for gaps
    area_km2 = float(catch_attrs().loc[gid, "area_calc"])
    # m3/s -> mm/day: q[m3/s] * 86400 s / (area_km2 * 1e6 m2) * 1000 mm/m
    mm = qobs * 86400.0 / (area_km2 * 1e6) * 1000.0
    mask = (mm.index.year >= YEAR_START) & (mm.index.year <= YEAR_END)
    return mm.loc[mask]


# Scale-independent regionalization attributes (the set that reached 0.42;
# baseflow_index deliberately EXCLUDED to avoid signature leakage).
REGIO_ATTRS = [
    "elev_mean",
    "p_mean",      # log-transformed in attr_matrix
    "frac_snow",
    "glac_fra",
    "area_calc",   # log-transformed (memory's 0.42 set used log-area, not slope)
    "aridity",
    "strm_dens",
]


def gauge_pool(degimpact=("u",), require_glac_max=None) -> list[int]:
    """Gauge ids with forcing + obs + attributes available in the window.

    Defaults to unregulated ('u') gauges — the honest evaluation denominator.
    """
    ga = gauge_attrs()
    ca = catch_attrs()
    ids = []
    for gid in ga.index:
        if ga.loc[gid, "degimpact"] not in degimpact:
            continue
        if gid not in ca.index:
            continue
        if require_glac_max is not None and float(ca.loc[gid, "glac_fra"]) > require_glac_max:
            continue
        if not (MET_DIR / f"ID_{gid}.csv").exists():
            continue
        if not (Q_DIR / f"ID_{gid}.csv").exists():
            continue
        ids.append(int(gid))
    return ids


if __name__ == "__main__":
    pool = gauge_pool()
    print(f"unregulated gauge pool: {len(pool)} gauges")
    gid = 9  # Syðri-Bægisá
    f = load_forcing(gid)
    o = load_obs_mm(gid)
    ov = o.dropna()
    print(f"gid {gid}: forcing {f.shape} {f.index.min().date()}..{f.index.max().date()}")
    print(f"  precip mean {f.precip.mean():.2f} mm/d, temp {f.temp.mean():.2f} C, pet {f.pet.mean():.2f}")
    print(f"  obs mm/d: n={ov.shape[0]} mean {ov.mean():.2f}  (runoff ratio {ov.mean()/f.precip.mean():.2f})")
    ca = catch_attrs().loc[gid]
    print(f"  area {ca.area_calc:.1f} km2, glac {ca.glac_fra:.2f}, elev {ca.elev_mean:.0f}")
