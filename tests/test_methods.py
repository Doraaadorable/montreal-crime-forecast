"""Checks on the pieces of the study that are easy to get subtly wrong."""
import json

import numpy as np
import pandas as pd

from calibrate import MIN_CALIBRATION, calibrate_group, conformal_quantile
from count_models import nb_dispersion
from data import MANIFEST_PATH, frozen_snapshot, incidents_from_snapshot
from hierarchy import build_hierarchy, projection, reconcile
from significance import block_bootstrap_ratio


def toy_bottom() -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols = pd.MultiIndex.from_product([["A", "B"], [1, 2, 3]])
    index = pd.date_range("2020-01-01", periods=24, freq="MS")
    return pd.DataFrame(rng.poisson(10, size=(24, 6)), index=index, columns=cols)


def test_hierarchy_aggregates_add_up():
    bottom = toy_bottom()
    all_series, S, level = build_hierarchy(bottom)
    assert S.shape == (1 + 2 + 3 + 6, 6)
    assert np.allclose(all_series["Total"], bottom.sum(axis=1))
    assert np.allclose(all_series["A"], bottom["A"].sum(axis=1))
    assert np.allclose(all_series["PDQ 2"], bottom.xs(2, axis=1, level=1).sum(axis=1))
    assert level.value_counts()["Category x district"] == 6


def test_reconciled_forecasts_are_coherent():
    _, S, _ = build_hierarchy(toy_bottom())
    Sm = S.to_numpy()
    base = np.random.default_rng(1).uniform(5, 50, size=len(S))  # deliberately incoherent
    for weights in (np.ones(len(S)), Sm.sum(axis=1)):
        P = projection(Sm, weights)
        coherent = P @ base
        assert np.allclose(coherent, Sm @ coherent[-S.shape[1]:])  # equals sum of its bottom
        assert np.allclose(P @ coherent, coherent)  # already-coherent forecasts are unchanged


def test_non_negative_forecasts_still_add_up():
    all_series, S, _ = build_hierarchy(toy_bottom())
    h = 2
    cutoff = all_series.index[-h - 1]
    future = all_series.index[-h:]
    rng = np.random.default_rng(5)
    rows = [{"unique_id": uid, "ds": ds, "cutoff": cutoff, "y": all_series.loc[ds, uid],
             "AutoETS": rng.normal(2, 6), "SeasonalNaive": 5.0}  # base forecasts go negative
            for uid in S.index for ds in future]
    rec = reconcile(pd.DataFrame(rows), all_series, S)
    n_bottom = S.shape[1]
    for method, grp in rec.groupby("method"):
        if method == "ETS base (not coherent)":
            continue
        assert (grp["forecast"] >= 0).all(), method
        wide = grp.pivot(index="unique_id", columns="ds", values="forecast").loc[S.index]
        assert np.allclose(wide.to_numpy(), S.to_numpy() @ wide.to_numpy()[-n_bottom:]), method


def test_conformal_quantile_uses_finite_sample_rank():
    scores = np.arange(1, 10) / 10  # 0.1 ... 0.9
    assert conformal_quantile(scores, 80) == 0.8  # ceil(10 * 0.8) = 8th smallest
    assert conformal_quantile(scores, 95) == 0.9  # rank capped at n


def test_calibration_ignores_outcomes_not_yet_observed():
    n, h = MIN_CALIBRATION + 20, 3
    cutoff = pd.date_range("2018-01-01", periods=n, freq="MS")
    rng = np.random.default_rng(2)
    grp = pd.DataFrame({"cutoff": cutoff, "ds": cutoff + pd.DateOffset(months=h),
                        "forecast": 100.0, "y": 100 + rng.normal(0, 10, n)})
    before = calibrate_group(grp)
    changed = grp.copy()
    row = n - 5
    # Outcomes for targets after this origin are unknown at the origin: changing them
    # must not move the interval issued at that origin.
    changed.loc[changed["ds"] > cutoff[row], "y"] += 500
    after = calibrate_group(changed)
    for col in ("q80_expanding", "q80_rolling36", "q80_aci"):
        assert before.loc[row, col] == after.loc[row, col]


def test_dispersion_is_near_zero_for_poisson_and_positive_when_overdispersed():
    rng = np.random.default_rng(3)
    mu = np.full(20000, 12.0)
    assert nb_dispersion(rng.poisson(mu), mu) < 0.01
    alpha = 0.25
    y = rng.negative_binomial(1 / alpha, 1 / (1 + alpha * mu))
    assert abs(nb_dispersion(y, mu) - alpha) < 0.03


def test_bootstrap_ratio_brackets_the_point_estimate():
    rng = np.random.default_rng(4)
    b = rng.uniform(5, 10, 77)
    ratio, lo, hi = block_bootstrap_ratio(b * 0.9, b)
    assert np.isclose(ratio, 0.9) and np.isclose(lo, 0.9) and np.isclose(hi, 0.9)
    ratio, lo, hi = block_bootstrap_ratio(b * rng.uniform(0.7, 1.1, 77), b)
    assert lo < ratio < hi


def test_frozen_snapshot_holds_every_incident_of_the_recorded_download():
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    incidents = incidents_from_snapshot(frozen_snapshot())
    assert len(incidents) == manifest["rows"]
    assert incidents["DATE"].min() == pd.Timestamp(manifest["first_incident_date"][:7] + "-01")
