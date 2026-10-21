#!/usr/bin/env python3
"""
test_unbinding.py - Residence times are exponential with mean
res_time_mean * site strength.

Static sites of three strengths (0.5, 1, 3) with res_time_mean = 1 s. For
every strength, the completed binding intervals must have
  - mean dt / p_off  (= res_time_mean * strength, up to dt/2)
  - a geometric (discrete exponential) distribution: KS test p > 0.01

    python tests/test_unbinding.py
"""

import os
import sys

import numpy as np
from scipy import stats

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.simulation import run_full_simulation
from smt_tracking.tracker import BindingTracker
from smt_tracking.events import dwell_table

DT = 0.01
TAU = 1.0
STRENGTHS = [0.5, 1.0, 3.0]


def main():
    tracker = BindingTracker(label_spec='random:1')
    run_full_simulation(
        num_display_frames=200, num_binding_sites=60, nucleus_diam_nm=2000,
        tf_concentration_nM=300, sim_dt_s=DT, display_dt_s=1.0,
        pct_free_init=100, pct_inter_init=0, pct_ns_init=0,
        d_free=0.31, d_free_std=0.03, d_inter=0.1, d_inter_std=0.05,
        d_ns=0.01, d_ns_std=0.005,
        bs_diff_mean=0.0, bs_diff_std=0.0, bs_jiggle_std=0.0, bs_hurst=0.25,
        k_free_to_inter=0.0, k_inter_to_free=0.0, k_free_to_ns=0.0,
        k_inter_to_ns=0.0, ns_res_time=0.5,
        cluster_distribution={1: 100}, res_time_mean=TAU,
        capture_radius_nm=50, k_on=np.inf,
        strength_multipliers=STRENGTHS, strength_percentages=[1, 1, 1],
        two_state_mode=True, tracker=tracker,
        site_rng=np.random.default_rng(1), mol_rng=np.random.default_rng(2))
    d = dwell_table(tracker.result())
    d = d[~d['censored']]
    ok = True
    for strength in STRENGTHS:
        dw = d.loc[np.isclose(d['site_strength'], strength), 'dwell_s'].values
        p_off = -np.expm1(-DT / (TAU * strength))
        expected = DT / p_off
        sem = dw.std(ddof=1) / np.sqrt(len(dw))
        steps = np.rint(dw / DT).astype(int)
        # KS against the geometric law of the number of steps (continuity-
        # corrected by comparing at half-integers)
        ks_p = stats.kstest(steps - 0.5, stats.geom(p_off).cdf).pvalue
        good = abs(dw.mean() - expected) < 4 * sem and ks_p > 0.01
        ok &= good
        print("%s strength %.1f: n=%5d  mean %.3f s (expected %.3f, s.e.m. %.3f)  "
              "CV %.2f (exp: 1)  KS p=%.2f" % (
                  "ok  " if good else "FAIL", strength, len(dw), dw.mean(),
                  expected, sem, dw.std() / dw.mean(), ks_p))
    print("exponential unbinding: %s" % ("passed" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
