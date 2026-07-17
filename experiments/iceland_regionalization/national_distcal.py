"""Scaled per-gauge DISTRIBUTED calibration over all 68 unregulated gauges.

Warm-starts each gauge at its lumped-calibrated params, calibrates uniform FUSE
params + Manning multiplier through its routed sub-model on clean 0.01 forcing
(2000-2003). Saves distributed-valid params -> donors for the LOO regionalization.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import distcal as dc
import lamah_io as io

CACHE = Path(__file__).parent / "cache"
OUT = CACHE / "distributed_calibrated.npz"


def main():
    d = np.load(CACHE / "lumped_freecal.npz", allow_pickle=True)
    lump = {int(g): d["vecs"][i] for i, g in enumerate(d["gids"])}
    pool = io.gauge_pool()
    ga = io.gauge_attrs()
    rows, vecs, manns, nsubs = [], [], [], []
    t0 = time.time()
    for i, gid in enumerate(pool):
        try:
            r = dc.calibrate(gid, warmstart_vec=lump.get(gid))
        except Exception as e:  # noqa: BLE001
            print(f"[{i+1}/{len(pool)}] gid {gid} FAILED: {e}")
            continue
        rows.append((r["gid"], r["kge"]))
        vecs.append(r["vec"]); manns.append(r["manning_mult"]); nsubs.append(r["n_sub"])
        med = np.median([x[1] for x in rows])
        print(f"[{i+1}/{len(pool)}] gid {gid:3d} {ga.loc[gid,'name'][:16]:16s} "
              f"n={r['n_sub']:4d} KGE={r['kge']:.3f} mann×{r['manning_mult']:.2f} "
              f"| median {med:.3f} | {time.time()-t0:.0f}s", flush=True)
        # checkpoint each iter (run is ~1h; keep partial results durable)
        np.savez(OUT, gids=np.array([x[0] for x in rows]),
                 kges=np.array([x[1] for x in rows]), vecs=np.array(vecs),
                 manning=np.array(manns), n_sub=np.array(nsubs),
                 cal_params=np.array(dc.lc.CAL_PARAMS))
    k = np.array([x[1] for x in rows])
    print(f"\nDISTRIBUTED in-sample median KGE = {np.median(k):.3f} "
          f"({np.sum(k>0.3)}/{len(k)} >0.3, {np.sum(k>0.5)} >0.5)")


if __name__ == "__main__":
    main()
