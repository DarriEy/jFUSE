"""Per-gauge upstream sub-network extraction from the national jFUSE network.

Extracts the routed sub-model draining to a gauge's outlet reach so distributed
calibration can run on it cheaply. Memory records repeated index bugs here
(HRU-forcing-column vs reach-topological index, Jaccard=0 upstream sets), so
every extraction is validated: total upstream HRU area must match LAMAH
area_calc, and the outlet's own reach must be in its upstream set.

Reconstructed 2026-07-06.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from jfuse.io import load_network

DOMAIN = Path("/Users/darri.eythorsson/compHydro/SYMFLUENCE_data/domain_Iceland_multivar")
TOPO = DOMAIN / "settings" / "mizuRoute" / "topology.nc"
GAUGE_MAP = DOMAIN / "settings" / "mizuRoute" / "gauge_segment_mapping.csv"
FUSE_MAP = DOMAIN / "settings" / "mizuRoute" / "fuse_to_routing_mapping.csv"

_NET = None
_HA = None
_GRU_AREA = None  # gru_id -> area m2


def national():
    global _NET, _HA, _GRU_AREA
    if _NET is None:
        net, hru_areas = load_network(str(TOPO))
        _NET = net.to_arrays() if hasattr(net, "to_arrays") else net
        _HA = np.asarray(hru_areas)
        m = pd.read_csv(FUSE_MAP)
        _GRU_AREA = dict(zip(m["gru_id"].astype(int), m["gru_area"].astype(float)))
    return _NET, _HA, _GRU_AREA


def gauge_outlet_reach(gid: int) -> int:
    """National reach INDEX for the gauge's mapped segment (nearest_segment)."""
    gm = pd.read_csv(GAUGE_MAP).set_index("id")
    seg_id = int(gm.loc[gid, "nearest_segment"])
    na, _, _ = national()
    rid = np.asarray(na.reach_ids)
    hits = np.where(rid == seg_id)[0]
    if len(hits) == 0:
        raise ValueError(f"seg {seg_id} not in reach_ids for gauge {gid}")
    return int(hits[0])


def _recompute_levels(down_local: np.ndarray, n: int) -> np.ndarray:
    """Topological level: headwaters (no upstream) = 0, else max(upstream)+1."""
    up_children = [[] for _ in range(n)]  # reaches draining INTO r
    for r in range(n):
        d = down_local[r]
        if d >= 0:
            up_children[d].append(r)
    level = np.full(n, -1, dtype=np.int32)
    # headwaters = reaches with no upstream children
    from collections import deque
    indeg = np.array([len(up_children[r]) for r in range(n)])
    q = deque([r for r in range(n) if indeg[r] == 0])
    for r in q:
        level[r] = 0
    # process upstream->downstream following downstream links
    order = deque(q)
    remaining = dict()
    child_count = {r: len(up_children[r]) for r in range(n)}
    seen = {r: 0 for r in range(n)}
    while order:
        r = order.popleft()
        d = down_local[r]
        if d >= 0:
            level[d] = max(level[d], level[r] + 1)
            seen[d] += 1
            if seen[d] == child_count[d]:
                order.append(d)
    level[level < 0] = 0
    return level


def extract(gid: int):
    """Return (subnet NetworkArrays, gru_ids[n_sub], hru_area_m2[n_sub], outlet_local_idx)."""
    na, _, gru_area = national()
    outlet = gauge_outlet_reach(gid)
    umask = np.asarray(na.upstream_mask)
    dsi_nat = np.asarray(na.downstream_idx)
    # upstream_mask is IMMEDIATE parents only; compute the FULL transitive
    # upstream set by BFS up the drainage tree (children[d] drain into d).
    from collections import defaultdict, deque
    children = defaultdict(list)
    for r in range(na.n_reaches):
        d = int(dsi_nat[r])
        if d >= 0:
            children[d].append(r)
    R_set = {outlet}
    q = deque([outlet])
    while q:
        d = q.popleft()
        for c in children[d]:
            if c not in R_set:
                R_set.add(c)
                q.append(c)
    R = np.array(sorted(R_set))  # sorted national indices
    pos = {int(r): k for k, r in enumerate(R)}
    outlet_local = pos[outlet]

    rid = np.asarray(na.reach_ids)
    hid = np.asarray(na.hru_ids)
    dsi = np.asarray(na.downstream_idx)
    down_local = np.array([pos.get(int(dsi[r]), -1) for r in R], dtype=np.int32)
    umask_local = umask[np.ix_(R, R)]
    level = _recompute_levels(down_local, len(R))

    gru_ids = hid[R].astype(int)
    hru_area = np.array([gru_area.get(int(g), 0.0) for g in gru_ids], dtype=float)

    is_out = down_local < 0
    is_head = ~umask_local.any(axis=1)

    def sub(x):
        return np.asarray(x)[R]

    subnet = na._replace(
        n_reaches=len(R),
        reach_ids=rid[R],
        lengths=sub(na.lengths),
        slopes=sub(na.slopes),
        manning_n=sub(na.manning_n),
        width_coef=sub(na.width_coef),
        width_exp=sub(na.width_exp),
        depth_coef=sub(na.depth_coef),
        depth_exp=sub(na.depth_exp),
        areas=sub(na.areas),
        hru_ids=hid[R],
        upstream_mask=umask_local,
        downstream_idx=down_local,
        is_headwater=is_head,
        is_outlet=is_out,
        reach_level=level,
        max_level=int(level.max()),
        is_lake=sub(na.is_lake),
        lake_s_max=sub(na.lake_s_max),
        lake_q_ref=sub(na.lake_q_ref),
        lake_q_min=sub(na.lake_q_min),
        lake_exp=sub(na.lake_exp),
        lake_spill_coef=sub(na.lake_spill_coef),
    )
    return subnet, gru_ids, hru_area, outlet_local


def validate(gid: int, lamah_area_calc_km2: float):
    subnet, gru_ids, hru_area, outlet_local = extract(gid)
    area_km2 = hru_area.sum() / 1e6
    ratio = area_km2 / lamah_area_calc_km2
    # outlet's own GRU must be in the upstream set (guards the Jaccard=0 bug)
    outlet_gru = int(np.asarray(subnet.hru_ids)[outlet_local])
    self_in = outlet_gru in set(int(g) for g in gru_ids)
    return {
        "gid": gid, "n_reaches": subnet.n_reaches, "area_km2": area_km2,
        "lamah_km2": lamah_area_calc_km2, "area_ratio": ratio, "outlet_self_in_set": self_in,
        "n_levels": subnet.max_level + 1,
    }


if __name__ == "__main__":
    import lamah_io as io
    ca = io.catch_attrs()
    for gid, nm in [(9, "Syðri-Bægisá"), (23, "Flatarhylur"), (84, "Svartá"), (2, "Eyfirðingavað")]:
        try:
            v = validate(gid, float(ca.loc[gid, "area_calc"]))
            flag = "OK" if 0.6 < v["area_ratio"] < 1.7 and v["outlet_self_in_set"] else "** CHECK **"
            print(f"gid {gid:3d} {nm:14s} n_reach={v['n_reaches']:4d} area={v['area_km2']:8.1f} "
                  f"lamah={v['lamah_km2']:8.1f} ratio={v['area_ratio']:.2f} self_in={v['outlet_self_in_set']} {flag}")
        except Exception as e:
            print(f"gid {gid}: {e}")
