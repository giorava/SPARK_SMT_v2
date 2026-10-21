#!/usr/bin/env python3
"""
test_physics.py - Consistency checks of the physics step.

Runs short simulations (two-state with cooperativity, three-state, static
and moving sites) with check_invariants=True, which after EVERY step asserts:
  - state 2 <-> bound site, and site occupancy points back to the molecule
  - per-cluster occupancy counts are up to date
  - bound molecules sit exactly on their site
  - no molecule is outside the nucleus
  - molecules released in a step neither bind nor switch in that step

    python tests/test_physics.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.simulation import run_full_simulation


BASE = dict(
    num_display_frames=20, num_binding_sites=30, nucleus_diam_nm=2000,
    tf_concentration_nM=300, sim_dt_s=0.01, display_dt_s=0.2,
    pct_free_init=80, pct_inter_init=40, pct_ns_init=20,
    d_free=0.31, d_free_std=0.03, d_inter=0.1, d_inter_std=0.05,
    d_ns=0.01, d_ns_std=0.005,
    bs_diff_mean=0.038, bs_diff_std=0.003, bs_jiggle_std=0.001, bs_hurst=0.25,
    k_free_to_inter=2.0, k_inter_to_free=3.0, k_free_to_ns=0.5,
    k_inter_to_ns=1.0, ns_res_time=0.5,
    cluster_distribution={1: 20, 5: 80}, cluster_radius_nm=100,
    res_time_mean=0.3,
    capture_radius_nm=80, k_on=150.0,
    num_synthetic_sites=6, synthetic_strength=1.0,
    synthetic_cluster_radius_nm=80, exposure_time_s=0.05,
)

CASES = {
    'two-state, on + off cooperativity': dict(two_state_mode=True, enable_cooperativity=True,
                                              coop_strength=2.0, coop_hill=1.0,
                                              coop_off_strength=1.0, coop_off_hill=2.0),
    'three-state': dict(two_state_mode=False),
    'static sites': dict(two_state_mode=True, bs_diff_mean=0.0, bs_diff_std=0.0,
                         bs_jiggle_std=0.0),
}


def main():
    for name, extra in CASES.items():
        for seed in range(3):
            kw = dict(BASE, **extra)
            run_full_simulation(**kw, check_invariants=True,
                                site_rng=np.random.default_rng([seed, 1]),
                                mol_rng=np.random.default_rng([seed, 2]))
        print("ok  %s (3 seeds)" % name)
    print("all physics checks passed")


if __name__ == '__main__':
    main()
