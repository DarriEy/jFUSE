"""Lumped jFUSE free-calibration on LAMAH catchment data (production config).

Calibrates the production 16-parameter set per catchment against LAMAH obs
(mm/d) using Adam through the differentiable FUSE model. This is the correctness
gate: the resulting attr-kNN LOO median should reproduce the memory's ~0.42.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

import jfuse.fuse.config as fcfg
from jfuse import ModelConfig, Parameters, PARAM_BOUNDS
from jfuse.fuse.model import fuse_simulate
from jfuse.fuse.state import State

import lamah_io as io

jax.config.update("jax_enable_x64", True)

DECISIONS = (
    "/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Iceland_multivar/"
    "settings/FUSE/fuse_zDecisions_iceland_large_domain_carra.txt"
)

# Production calibration set (config_jfuse_prod6_18yr.yaml JFUSE_PARAMS_TO_CALIBRATE).
CAL_PARAMS = [
    "S1_max", "S2_max", "ku", "ki", "ks", "n", "Ac_max", "b",
    "f_rchr", "T_rain", "T_melt", "MFMAX", "MFMIN", "smooth_frac", "SCF", "opg",
]

LAPSE_C_PER_M = 0.0065  # snow/melt lapse for glacier surface dtemp


def prod_config() -> ModelConfig:
    dec = fcfg.parse_decisions_file(DECISIONS)
    cfg = fcfg.config_from_decisions(dec)
    return cfg._replace(enable_glacier=True)


_LO = jnp.array([PARAM_BOUNDS[p][0] for p in CAL_PARAMS])
_HI = jnp.array([PARAM_BOUNDS[p][1] for p in CAL_PARAMS])


def _unbounded_to_params(z):
    """Map unbounded z (len 16) -> bounded values via sigmoid."""
    return _LO + (_HI - _LO) * jax.nn.sigmoid(z)


def _bounded_to_unbounded(vals):
    frac = np.clip((np.asarray(vals) - np.asarray(_LO)) / (np.asarray(_HI) - np.asarray(_LO)), 1e-4, 1 - 1e-4)
    return np.log(frac / (1 - frac))


def build_params(bounded_vec):
    """Return a Parameters pytree (n_hrus=1) with CAL_PARAMS set to bounded_vec."""
    p = Parameters.default(n_hrus=1)
    p = eqx.tree_at(
        lambda t: [getattr(t, k) for k in CAL_PARAMS],
        p,
        [jnp.atleast_1d(bounded_vec[i]) for i in range(len(CAL_PARAMS))],
    )
    return p


def kge(sim, obs):
    s = np.asarray(sim); o = np.asarray(obs)
    m = np.isfinite(o) & np.isfinite(s)
    s, o = s[m], o[m]
    if s.size < 30 or o.std() == 0:
        return -9.99
    r = np.corrcoef(s, o)[0, 1]
    alpha = s.std() / o.std()
    beta = s.mean() / o.mean()
    return 1.0 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def _kge_loss_jax(sim, obs, valid):
    """Differentiable KGE loss (masked). valid: bool array, obs NaN->0."""
    n = jnp.sum(valid)
    om = jnp.sum(obs * valid) / n
    sm = jnp.sum(sim * valid) / n
    os = jnp.sqrt(jnp.sum(valid * (obs - om) ** 2) / n)
    ss = jnp.sqrt(jnp.sum(valid * (sim - sm) ** 2) / n)
    cov = jnp.sum(valid * (obs - om) * (sim - sm)) / n
    r = cov / (os * ss + 1e-8)
    alpha = ss / (os + 1e-8)
    beta = sm / (om + 1e-8)
    return jnp.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)


def _make_simulate(cfg):
    @jax.jit
    def sim_runoff(bounded_vec, forcing, glac_frac, glac_dtemp, s0):
        p = build_params(bounded_vec)
        runoff, _ = fuse_simulate(
            forcing, s0, p, cfg, 1.0, 1,
            glacier_frac=glac_frac, glacier_dtemp=glac_dtemp,
        )
        return runoff[:, 0]
    return sim_runoff


def _make_grad(cfg):
    """Module-level jitted value_and_grad that takes ALL gauge data as args, so
    it compiles ONCE (shapes are identical across gauges) and is reused for every
    gauge. Previously grad_fn was a per-gauge closure -> recompiled each call."""
    sim = _make_simulate(cfg)

    def loss_fn(z, precip, pet, temp, glac_frac, glac_dtemp, s0, obs_j, valid_j):
        vec = _unbounded_to_params(z)
        simq = sim(vec, (precip, pet, temp), glac_frac, glac_dtemp, s0)
        return _kge_loss_jax(simq, obs_j, valid_j)

    return sim, jax.jit(jax.value_and_grad(loss_fn))


def calibrate_gauge(gid, cfg, sim_runoff, n_steps=400, lr=0.05, seed=0, precip_col="prec_carra", grad_fn=None):
    f = io.load_forcing(gid, precip_col=precip_col)
    obs = io.load_obs_mm(gid).reindex(f.index)
    ca = io.catch_attrs().loc[gid]

    precip = jnp.asarray(f["precip"].values)[:, None]
    pet = jnp.asarray(f["pet"].values)[:, None]
    temp = jnp.asarray(f["temp"].values)[:, None]
    forcing = (precip, pet, temp)

    glac_frac = jnp.atleast_1d(jnp.asarray(float(ca["glac_fra"])))
    g_el = float(ca["g_mean_el"]) if np.isfinite(ca["g_mean_el"]) else float(ca["elev_mean"])
    dtemp = -(g_el - float(ca["elev_mean"])) * LAPSE_C_PER_M  # glacier colder if higher
    glac_dtemp = jnp.atleast_1d(jnp.asarray(dtemp))
    s0 = State.default(1)

    obs_np = obs.values.astype(float)
    valid_np = np.isfinite(obs_np)
    valid_np[: io.WARMUP_DAYS] = False  # exclude warmup year
    obs_j = jnp.asarray(np.nan_to_num(obs_np))
    valid_j = jnp.asarray(valid_np.astype(float))

    if grad_fn is None:
        _, grad_fn = _make_grad(cfg)

    # Warm start at bounds midpoint (sigmoid(0)).
    z = jnp.zeros(len(CAL_PARAMS))
    opt = optax.adam(lr)
    state = opt.init(z)
    best_loss, best_z = np.inf, z
    for i in range(n_steps):
        l, g = grad_fn(z, precip, pet, temp, glac_frac, glac_dtemp, s0, obs_j, valid_j)
        g = jnp.nan_to_num(g)
        updates, state = opt.update(g, state)
        z = optax.apply_updates(z, updates)
        if float(l) < best_loss:
            best_loss, best_z = float(l), z

    vec = _unbounded_to_params(best_z)
    sim = np.asarray(sim_runoff(vec, forcing, glac_frac, glac_dtemp, s0))
    score = kge(sim[valid_np], obs_np[valid_np])
    return {
        "gid": int(gid),
        "kge": float(score),
        "params": {k: float(np.asarray(vec)[j]) for j, k in enumerate(CAL_PARAMS)},
        "vec": np.asarray(vec),
    }


if __name__ == "__main__":
    import sys
    cfg = prod_config()
    print("config:", cfg.upper_arch, cfg.lower_arch, cfg.surface_runoff, "glacier", cfg.enable_glacier)
    sim_runoff, grad_fn = _make_grad(cfg)
    test_gids = [int(x) for x in sys.argv[1:]] or [9, 23, 84]  # Syðri-Bægisá, Flatarhylur, Svartá
    for gid in test_gids:
        r = calibrate_gauge(gid, cfg, sim_runoff, grad_fn=grad_fn)
        nm = io.gauge_attrs().loc[gid, "name"]
        print(f"gid {gid:3d} {nm:20s} calibrated KGE = {r['kge']:.3f}")
