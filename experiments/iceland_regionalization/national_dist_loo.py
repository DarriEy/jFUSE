"""THE DECISIVE TEST: do DISTRIBUTED-calibrated params regionalize (leave-one-out)?

For each held-out gauge, transfer the distributed-calibrated FUSE params of its
attribute-nearest donor(s) onto its OWN routed sub-network + clean 0.01 forcing,
run the distributed model, and score at the outlet. Compares:
  - old buggy pipeline distributed-LOO: -0.06
  - lumped-LOO ceiling (this rebuild): 0.426
  - distributed in-sample (this rebuild): ~0.6-0.7

Reconstructed 2026-07-06.
"""
from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

import distcal as dc
import lamah_io as io
import lumped_cal as lc
from regionalize_loo import attr_matrix

CACHE = Path(__file__).parent / "cache"


def main(k=5):
    d = np.load(CACHE / "distributed_calibrated.npz", allow_pickle=True)
    all_gids = [int(g) for g in d["gids"]]
    all_vecs = {g: d["vecs"][i] for i, g in enumerate(all_gids)}
    all_kge = dict(zip(all_gids, d["kges"]))
    # Restrict to gauges with adequate obs in 2000-2003 AND a valid calibration
    # (drop the ~21 no-obs gauges whose params are garbage as donors).
    usable = set(int(x) for x in np.load(CACHE / "usable_gids.npy"))
    gids = [g for g in all_gids if g in usable and all_kge[g] > -1.0]
    vecs = np.array([all_vecs[g] for g in gids])
    insample = {g: all_kge[g] for g in gids}
    print(f"usable distributed pool: {len(gids)} gauges (of {len(all_gids)} calibrated)")
    Xs = attr_matrix(np.array(gids))
    st = dc._load()

    # Pre-build each gauge's sub-model + a no-grad forward sim once (compiles per gauge).
    print("building gauge sub-models...")
    simfns, gdata = {}, {}
    for gid in gids:
        g = dc.build_gauge(gid)
        sim_q, _ = dc.make_loss(g, st["cfg"])
        simfns[gid] = sim_q
        gdata[gid] = g

    knn_kge, nn1_kge = [], []
    for i, gid in enumerate(gids):
        dist = np.sqrt(((Xs - Xs[i]) ** 2).sum(1)); dist[i] = np.inf
        order = np.argsort(dist)
        g = gdata[gid]; sim_q = simfns[gid]
        v = np.asarray(g["valid"]).astype(bool); o = np.asarray(g["obs"])
        # donor param -> forward run on THIS gauge's sub-network (manning mult=1)
        def run(donor_idx):
            z = jnp.asarray(lc._bounded_to_unbounded(vecs[donor_idx]))
            return np.asarray(sim_q(z, jnp.array(0.0)))
        nn1_kge.append(lc.kge(run(order[0])[v], o[v]))
        sims = [run(j) for j in order[:k]]
        ens = np.mean(sims, axis=0)
        knn_kge.append(lc.kge(ens[v], o[v]))
        if i % 10 == 0 or i == len(gids) - 1:
            print(f"  LOO [{i+1}/{len(gids)}] gid {gid} in={insample[gid]:.2f} "
                  f"1NN={nn1_kge[-1]:.2f} kNN={knn_kge[-1]:.2f}", flush=True)

    knn = np.array(knn_kge); nn1 = np.array(nn1_kge)
    np.savez(CACHE / "distributed_loo.npz", gids=gids, knn=knn, nn1=nn1)
    print("\n" + "=" * 60)
    print("DISTRIBUTED-PARAM REGIONALIZATION (leave-one-out)")
    print("=" * 60)
    print(f"  distributed in-sample median : {np.median([insample[g] for g in gids]):.3f}  (n={len(gids)})")
    print(f"  attr-1NN LOO median          : {np.median(nn1):.3f}  ({np.sum(nn1>0.3)}/{len(nn1)} >0.3)")
    print(f"  kNN-ens(k={k}) LOO median      : {np.median(knn):.3f}  ({np.sum(knn>0.3)}/{len(knn)} >0.3, {np.sum(knn>0.5)} >0.5)")
    print(f"\n  REFERENCE: old buggy pipeline distributed-LOO = -0.06")
    print(f"             lumped-LOO ceiling (this rebuild)  =  0.426")


if __name__ == "__main__":
    main()
