"""Full lumped free-cal (68 gauges) + attr-kNN leave-one-out regionalization.

Correctness gate: in-sample median should be ~0.59 and kNN-ensemble LOO ~0.42
(memory's clean-LAMAH result). Regenerates the lost free-cal + attribute caches.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import lamah_io as io
import lumped_cal as lc
from jfuse.fuse.state import State

CACHE = Path(__file__).parent / "cache"
CACHE.mkdir(exist_ok=True)
FREECAL = CACHE / "lumped_freecal.npz"


def run_freecal(precip_col="prec_carra"):
    cfg = lc.prod_config()
    sim_runoff, grad_fn = lc._make_grad(cfg)
    pool = io.gauge_pool()  # 68 unregulated
    rows, vecs = [], []
    t0 = time.time()
    for i, gid in enumerate(pool):
        r = lc.calibrate_gauge(gid, cfg, sim_runoff, precip_col=precip_col, grad_fn=grad_fn)
        rows.append((r["gid"], r["kge"]))
        vecs.append(r["vec"])
        if i % 5 == 0 or i == len(pool) - 1:
            med = np.median([x[1] for x in rows])
            print(f"[{i+1}/{len(pool)}] gid {gid} kge {r['kge']:.3f} | running median {med:.3f} | {time.time()-t0:.0f}s")
    gids = np.array([x[0] for x in rows])
    kges = np.array([x[1] for x in rows])
    vecs = np.array(vecs)
    np.savez(FREECAL, gids=gids, kges=kges, vecs=vecs, cal_params=np.array(lc.CAL_PARAMS))
    print(f"\nIN-SAMPLE free-cal median KGE = {np.median(kges):.3f} "
          f"({np.sum(kges>0.3)}/{len(kges)} >0.3, {np.sum(kges>0.5)}/{len(kges)} >0.5)")
    return gids, kges, vecs


def attr_matrix(gids):
    ca = io.catch_attrs()
    X = np.array([[float(ca.loc[g, a]) for a in io.REGIO_ATTRS] for g in gids])
    # log-transform heavy-tailed attrs (p_mean, area_calc); standardize all
    for name in ("p_mean", "area_calc"):
        if name in io.REGIO_ATTRS:
            j = io.REGIO_ATTRS.index(name)
            X[:, j] = np.log(np.clip(X[:, j], 1e-3, None))
    mu, sd = X.mean(0), X.std(0) + 1e-9
    return (X - mu) / sd


def loo_regionalize(gids, vecs, k=5):
    """Leave-one-out attr-kNN output-ensemble + 1-NN param transfer."""
    cfg = lc.prod_config()
    sim_runoff = lc._make_simulate(cfg)
    Xs = attr_matrix(gids)
    s0 = State.default(1)

    # Precompute target forcing/obs/glacier
    forc, obsv, valid, glac = {}, {}, {}, {}
    for gid in gids:
        f = io.load_forcing(gid)
        o = io.load_obs_mm(gid).reindex(f.index).values.astype(float)
        v = np.isfinite(o); v[: io.WARMUP_DAYS] = False
        forc[gid] = (jnp.asarray(f["precip"].values)[:, None],
                     jnp.asarray(f["pet"].values)[:, None],
                     jnp.asarray(f["temp"].values)[:, None])
        obsv[gid] = o; valid[gid] = v
        ca = io.catch_attrs().loc[gid]
        g_el = float(ca["g_mean_el"]) if np.isfinite(ca["g_mean_el"]) else float(ca["elev_mean"])
        glac[gid] = (jnp.atleast_1d(jnp.asarray(float(ca["glac_fra"]))),
                     jnp.atleast_1d(jnp.asarray(-(g_el - float(ca["elev_mean"])) * lc.LAPSE_C_PER_M)))

    knn_kge, nn1_kge = [], []
    for i, gid in enumerate(gids):
        d = np.sqrt(((Xs - Xs[i]) ** 2).sum(1))
        d[i] = np.inf
        order = np.argsort(d)
        gf, gd = glac[gid]
        # 1-NN param transfer
        sim1 = np.asarray(sim_runoff(jnp.asarray(vecs[order[0]]), forc[gid], gf, gd, s0))
        nn1_kge.append(lc.kge(sim1[valid[gid]], obsv[gid][valid[gid]]))
        # k-NN output ensemble
        sims = []
        for j in order[:k]:
            sims.append(np.asarray(sim_runoff(jnp.asarray(vecs[j]), forc[gid], gf, gd, s0)))
        ens = np.mean(sims, axis=0)
        knn_kge.append(lc.kge(ens[valid[gid]], obsv[gid][valid[gid]]))
        if i % 10 == 0:
            print(f"  LOO [{i+1}/{len(gids)}] gid {gid} 1NN {nn1_kge[-1]:.2f} kNN {knn_kge[-1]:.2f}")

    knn = np.array(knn_kge); nn1 = np.array(nn1_kge)
    np.savez(CACHE / "lumped_loo.npz", gids=gids, knn=knn, nn1=nn1)
    print("\n=== LUMPED REGIONALIZATION (LOO) ===")
    print(f"attr-1NN   median KGE = {np.median(nn1):.3f}  ({np.sum(nn1>0.3)}/{len(nn1)} >0.3)")
    print(f"kNN-ens(k={k}) median = {np.median(knn):.3f}  ({np.sum(knn>0.3)}/{len(knn)} >0.3, {np.sum(knn>0.5)} >0.5)")
    print(f"  [gate: memory clean-LAMAH kNN-ens ~0.42]")
    return knn, nn1


if __name__ == "__main__":
    if FREECAL.exists():
        d = np.load(FREECAL, allow_pickle=True)
        gids, kges, vecs = d["gids"], d["kges"], d["vecs"]
        print(f"loaded cached free-cal: median {np.median(kges):.3f}, {len(gids)} gauges")
    else:
        gids, kges, vecs = run_freecal()
    loo_regionalize(gids, vecs)
