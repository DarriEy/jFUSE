"""Assemble the clean 0.01 national per-GRU distributed forcing (2000-2003).

The distributed worker's FUSE_input is stale 0.025 (Feb build); the clean 0.01
forcing only exists as monthly basin_averaged_data files. This module assembles
them into [T, n_gru] precip/temp arrays keyed by GRU_ID (hruId == gru_id), plus
Hamon PET, and caches the result. Used to build per-gauge distributed sub-models.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

DOMAIN = Path("/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Iceland_multivar")
BAV = DOMAIN / "data" / "forcing" / "basin_averaged_data"
CACHE = Path(__file__).parent / "cache"
NAT_CACHE = CACHE / "national_forcing_0p01_2000_2003.npz"


def _hamon_pet(temp_c: np.ndarray, lat_deg: np.ndarray, doy: np.ndarray) -> np.ndarray:
    """Hamon PET (mm/day). temp_c [T,G], lat_deg [G], doy [T]."""
    lat = np.deg2rad(lat_deg)[None, :]
    decl = 0.4093 * np.sin(2 * np.pi * doy / 365.0 - 1.405)[:, None]
    ws = np.arccos(np.clip(-np.tan(lat) * np.tan(decl), -1, 1))
    daylight = 24.0 * ws / np.pi  # hours
    es = 6.108 * np.exp(17.27 * temp_c / (temp_c + 237.3))  # sat vapor pressure mb
    rhosat = 216.7 * es / (temp_c + 273.3)  # g/m3
    pet = 0.1651 * (daylight / 12.0) * rhosat  # mm/day (standard Hamon)
    return np.clip(pet, 0.0, None)


def build(force=False):
    if NAT_CACHE.exists() and not force:
        d = np.load(NAT_CACHE, allow_pickle=True)
        return d["precip"], d["pet"], d["temp"], d["gru_ids"], pd.to_datetime(d["time"])

    files = sorted(glob.glob(str(BAV / "*CARRA_remapped_*.nc")))
    print(f"assembling {len(files)} monthly files (3-hourly -> daily)...")
    # hruId/lat are static per HRU: read from ONE file (open_mfdataset mangles
    # hruId by concatenating it along the time axis).
    ds0 = xr.open_dataset(files[0])
    hruId = ds0["hruId"].values.astype(int)  # column -> gru_id
    lat = ds0["latitude"].values
    if lat.ndim > 1:
        lat = lat[0]
    ds0.close()
    ds = xr.open_mfdataset(files, combine="by_coords")
    # Forcing is 3-hourly; resample to daily. Precip flux (kg m-2 s-1): daily
    # total mm = daily-mean rate * 86400. Temp: daily mean.
    precip_d = ds["precipitation_flux"].resample(time="1D").mean()
    temp_d = ds["air_temperature"].resample(time="1D").mean()
    time = pd.to_datetime(precip_d["time"].values)
    precip = (precip_d.values * 86400.0).astype(np.float32)  # mm/day
    temp = (temp_d.values - 273.15).astype(np.float32)  # C
    ds.close()

    doy = time.dayofyear.values.astype(float)
    pet = _hamon_pet(temp.astype(float), lat.astype(float), doy).astype(np.float32)

    np.savez(
        NAT_CACHE, precip=precip, pet=pet, temp=temp,
        gru_ids=hruId, lat=lat, time=time.values,
    )
    print(f"cached: precip {precip.shape}, {time[0].date()}..{time[-1].date()}")
    return precip, pet, temp, hruId, time


def gru2col(gru_ids: np.ndarray) -> dict:
    return {int(g): j for j, g in enumerate(gru_ids)}


if __name__ == "__main__":
    precip, pet, temp, gru_ids, time = build(force=True)
    print(f"national forcing: {precip.shape} (T x G), {time[0].date()}..{time[-1].date()}")
    print(f"  precip mean {precip.mean():.2f} mm/d, temp {temp.mean():.2f} C, pet {pet.mean():.2f} mm/d")
    print(f"  gru_ids: {len(gru_ids)} cols, range {gru_ids.min()}..{gru_ids.max()}, monotonic={bool(np.all(np.diff(gru_ids)>=0))}")
