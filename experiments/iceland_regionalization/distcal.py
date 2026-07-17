"""Per-gauge DISTRIBUTED (routed) calibration on clean 0.01 national forcing.

For each gauge: extract its upstream sub-network, build per-GRU sub-forcing from
the national 0.01 arrays, warm-start FUSE params at the gauge's lumped-calibrated
vector, and calibrate uniform FUSE params + a Manning-n multiplier through the
routed sub-model against gauge discharge (m3/s). These distributed-valid params
are the donors for the decisive LOO regionalization test.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd

import jfuse.fuse.config as fcfg
from jfuse import ModelConfig, Parameters
from jfuse.coupled import coupled_simulate_full
from jfuse.fuse.state import State as FUSEState

import lamah_io as io
import lumped_cal as lc
import national_forcing as nf
import subnet as sn

jax.config.update("jax_enable_x64", True)

DOMAIN = sn.DOMAIN
GLAC_FRAC = DOMAIN / "settings" / "JFUSE" / "glacier_fraction_by_gru.csv"
GLAC_DTEMP = DOMAIN / "settings" / "JFUSE" / "glacier_dtemp_by_gru.csv"
Q_DIR = io.Q_DIR

_STATE = {}


def _load():
    if _STATE:
        return _STATE
    precip, pet, temp, gru_ids, time = nf.build()
    _STATE.update(
        precip=precip, pet=pet, temp=temp, time=time,
        g2c=nf.gru2col(gru_ids),
        cfg=lc.prod_config(),
        gfrac=dict(zip(*[pd.read_csv(GLAC_FRAC)[c] for c in ("gru_id", "glacier_fraction")])),
        gdtemp=dict(zip(*[pd.read_csv(GLAC_DTEMP)[c] for c in ("gru_id", "glacier_dtemp")])),
        warmup=io.WARMUP_DAYS,
    )
    return _STATE


def load_obs_cms(gid: int, time_index) -> np.ndarray:
    """Gauge discharge (m3/s) aligned to the national forcing daily index."""
    q = pd.read_csv(Q_DIR / f"ID_{gid}.csv", sep=";")
    q.index = pd.to_datetime(dict(year=q["YYYY"], month=q["MM"], day=q["DD"]))
    s = q["qobs"].astype(float)
    s[s < 0] = np.nan
    return s.reindex(time_index).values


def build_gauge(gid: int):
    """Assemble everything needed to calibrate one gauge's sub-model."""
    st = _load()
    subnet, gru_ids_sub, hru_area, outlet_local = sn.extract(gid)
    # 6 network reaches are orphans (GRU has no forcing); they carry gru_area=0
    # so contribute zero lateral inflow regardless. Map them to placeholder col 0.
    cols = np.array([st["g2c"].get(int(g), 0) for g in gru_ids_sub])
    precip = jnp.asarray(st["precip"][:, cols])
    pet = jnp.asarray(st["pet"][:, cols])
    temp = jnp.asarray(st["temp"][:, cols])
    gf = jnp.asarray([st["gfrac"].get(int(g), 0.0) for g in gru_ids_sub])
    gd = jnp.asarray([st["gdtemp"].get(int(g), 0.0) for g in gru_ids_sub])
    reach_hru_col = jnp.arange(len(gru_ids_sub))  # sub-forcing col i == sub-reach i
    hru_areas = jnp.asarray(hru_area)
    manning0 = jnp.asarray(np.asarray(subnet.manning_n))
    obs = load_obs_cms(gid, st["time"])
    valid = np.isfinite(obs)
    valid[: st["warmup"]] = False
    return dict(
        subnet=subnet, precip=precip, pet=pet, temp=temp, gf=gf, gd=gd,
        reach_hru_col=reach_hru_col, hru_areas=hru_areas, manning0=manning0,
        outlet_local=outlet_local, obs=jnp.asarray(np.nan_to_num(obs)),
        valid=jnp.asarray(valid.astype(float)), n_valid=int(valid.sum()),
        n_sub=len(gru_ids_sub),
    )


def _params_from_vec(vec):
    """Uniform FUSE params (shape (1,) broadcast over sub-HRUs) from 16-vec."""
    p = Parameters.default(n_hrus=1)
    return eqx.tree_at(
        lambda t: [getattr(t, k) for k in lc.CAL_PARAMS],
        p, [jnp.atleast_1d(vec[i]) for i in range(len(lc.CAL_PARAMS))],
    )


def make_loss(g, cfg):
    """Jitted value_and_grad over (fuse_z[16], manning_log) for one gauge."""
    lo, hi = lc._LO, lc._HI
    subnet, outlet = g["subnet"], g["outlet_local"]
    forcing = (g["precip"], g["pet"], g["temp"])
    s0 = FUSEState.default(g["n_sub"])

    def sim_q(fuse_z, manning_log):
        vec = lo + (hi - lo) * jax.nn.sigmoid(fuse_z)
        params = _params_from_vec(vec)
        manning = g["manning0"] * jnp.exp(manning_log)
        _, Q_all, _ = coupled_simulate_full(
            forcing, params, manning, subnet, g["hru_areas"], cfg,
            initial_fuse_state=s0, glacier_frac=g["gf"], glacier_dtemp=g["gd"],
            reach_hru_col=g["reach_hru_col"],
        )
        return Q_all[:, outlet]

    def loss(fuse_z, manning_log):
        q = sim_q(fuse_z, manning_log)
        return lc._kge_loss_jax(q, g["obs"], g["valid"])

    return sim_q, jax.jit(jax.value_and_grad(loss, argnums=(0, 1)))


def calibrate(gid, warmstart_vec=None, n_steps=180, lr=0.05):
    st = _load()
    g = build_gauge(gid)
    sim_q, vg = make_loss(g, st["cfg"])
    # warm start at lumped-calibrated params (bounded->unbounded)
    if warmstart_vec is not None:
        fuse_z = jnp.asarray(lc._bounded_to_unbounded(warmstart_vec))
    else:
        fuse_z = jnp.zeros(len(lc.CAL_PARAMS))
    mlog = jnp.array(0.0)
    opt = optax.adam(lr)
    ostate = opt.init((fuse_z, mlog))
    best = (np.inf, fuse_z, mlog)
    for i in range(n_steps):
        l, (gz, gm) = vg(fuse_z, mlog)
        gz = jnp.nan_to_num(gz); gm = jnp.nan_to_num(gm)
        upd, ostate = opt.update((gz, gm), ostate)
        fuse_z, mlog = optax.apply_updates((fuse_z, mlog), upd)
        if float(l) < best[0]:
            best = (float(l), fuse_z, mlog)
    _, bz, bm = best
    q = np.asarray(sim_q(bz, bm))
    v = np.asarray(g["valid"]).astype(bool)
    o = np.asarray(g["obs"])
    score = lc.kge(q[v], o[v])
    vec = np.asarray(lc._LO + (lc._HI - lc._LO) * jax.nn.sigmoid(bz))
    return {"gid": int(gid), "kge": float(score), "vec": vec,
            "manning_mult": float(np.exp(float(bm))), "n_sub": g["n_sub"]}


if __name__ == "__main__":
    import sys, time
    d = np.load(Path(__file__).parent / "cache" / "lumped_freecal.npz", allow_pickle=True)
    lump = {int(gg): d["vecs"][i] for i, gg in enumerate(d["gids"])}
    gids = [int(x) for x in sys.argv[1:]] or [9, 42, 23, 84, 2]
    for gid in gids:
        t0 = time.time()
        r = calibrate(gid, warmstart_vec=lump.get(gid))
        nm = io.gauge_attrs().loc[gid, "name"]
        print(f"gid {gid:3d} {nm:16s} n_sub={r['n_sub']:4d} DIST-CAL KGE={r['kge']:.3f} "
              f"mann×{r['manning_mult']:.2f} ({time.time()-t0:.0f}s)")
