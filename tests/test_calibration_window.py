"""Tests for the calibration window and warmup alignment in the gradient loss.

Two defects are guarded here, both of which affected only the paths used by
gradient-based optimizers (ADAM, L-BFGS with native gradients):

1. The loss dropped warmup and then scored everything that remained. With a
   calibration/evaluation split configured that span covers both windows, so
   the optimizer trained on the held-out evaluation period.

2. The single-gauge loss advanced the simulation past warmup but truncated
   the observations from index 0 (``obs[: len(sim_eval)]``), comparing
   simulated day ``t + warmup`` against observed day ``t``.
"""

import numpy as np
import pytest

jnp = pytest.importorskip("jax.numpy")

from jfuse.calibration.worker import multi_gauge_kge_loss

WARMUP = 50
N = 400
G = 3


def _series():
    """Simulated discharge per segment, plus observations that match it."""
    rng = np.random.default_rng(0)
    t = np.arange(N)
    base = 5.0 + 3.0 * np.sin(t * 2 * np.pi / 120.0) + rng.normal(0, 0.3, N)
    Q_all = np.column_stack([base * (1.0 + 0.1 * g) for g in range(G)])
    # Observations equal the simulation at the same timestamps.
    gauge_obs = Q_all.copy()
    return jnp.array(Q_all), jnp.array(gauge_obs)


def test_aligned_series_scores_near_perfect():
    """Identical sim and obs must score KGE ~ 1 (loss ~ 0).

    This is the alignment guard: if either side were offset by warmup, a
    seasonal signal would decorrelate and the loss would be far from zero.
    """
    Q_all, gauge_obs = _series()
    loss = float(multi_gauge_kge_loss(Q_all, list(range(G)), gauge_obs, WARMUP))
    assert loss == pytest.approx(0.0, abs=1e-6)


def test_offset_observations_score_badly():
    """A warmup-sized offset must be visible as a much worse loss.

    Pins that the test above is actually sensitive to misalignment.
    """
    Q_all, gauge_obs = _series()
    shifted = jnp.concatenate([gauge_obs[WARMUP:], gauge_obs[:WARMUP]], axis=0)
    loss = float(multi_gauge_kge_loss(Q_all, list(range(G)), shifted, WARMUP))
    assert loss > 0.1


def test_full_span_slice_is_a_noop():
    """A slice covering the whole post-warmup record must not change the loss."""
    Q_all, gauge_obs = _series()
    unsliced = float(multi_gauge_kge_loss(Q_all, list(range(G)), gauge_obs, WARMUP))
    full = float(
        multi_gauge_kge_loss(Q_all, list(range(G)), gauge_obs, WARMUP, cal_slice=(0, N - WARMUP))
    )
    assert full == pytest.approx(unsliced, abs=1e-9)


def test_narrower_slice_is_applied():
    """Restricting to a sub-window must actually change what is scored."""
    Q_all, gauge_obs = _series()
    # Corrupt the back half of the observations; a calibration slice covering
    # only the front half must not see it.
    span = N - WARMUP
    obs = np.array(gauge_obs)
    obs[WARMUP + span // 2 :, :] *= 5.0
    obs_j = jnp.array(obs)

    front = float(
        multi_gauge_kge_loss(Q_all, list(range(G)), obs_j, WARMUP, cal_slice=(0, span // 2))
    )
    whole = float(multi_gauge_kge_loss(Q_all, list(range(G)), obs_j, WARMUP))

    # The front half is still a clean match; the whole record is not.
    assert front == pytest.approx(0.0, abs=1e-6)
    assert whole > front + 0.1
