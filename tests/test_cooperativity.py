#!/usr/bin/env python3
"""
test_cooperativity.py - Binding and unbinding probabilities per step.

A cluster of 6 sites (plus one lone site) is set up by hand with n of its
sites occupied. Repeating a single physics stage many times, the measured
probability that one molecule binds (unbinds) in a step must equal
    binding    1 - exp(-k_on  * f_on(n)       * dt),  n = bound in cluster
    unbinding  1 - exp(-k_off / f_off(n - 1)  * dt),  n - 1 = OTHER bound
with f(n) = 1 + (strength * n) ** hill and f = 1 for a lone site.
Also checks that k_on_prob converts to the same per-step probability.

    python tests/test_cooperativity.py
"""

import os
import sys

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.physics import SimState, StepParams, bind, unbind, coop_factor
from spark.params import binding_rate

DT = 0.01
K_ON = 30.0                 # 1/s  -> p_on(0) = 0.26
TAU = 0.05                  # s    -> k_off = 20 /s, p_off(0) = 0.18
ON = dict(strength=0.5, hill=1.5)
OFF = dict(strength=0.8, hill=2.0)
TRIALS = 6000

# 6 sites of cluster 0 on a ring (300 nm apart) + 1 lone site (cluster 1)
SITES = np.array([[0.3 * np.cos(a), 0.3 * np.sin(a), 0.0]
                  for a in np.linspace(0, 2 * np.pi, 6, endpoint=False)]
                 + [[0.0, 0.0, 0.6]])
CLUSTER = np.array([0, 0, 0, 0, 0, 0, 1])


def params():
    return StepParams(
        sim_dt_s=DT, radius_um=1.0, inner_radius=0.95, capture_radius_um=0.05,
        site_strengths=np.ones(len(SITES)), site_cluster_idx=CLUSTER,
        cluster_D=np.zeros(2), num_clusters=2,
        bs_diff_mean=0.0, bs_diff_std=0.0, bs_jiggle_std=0.0, bs_hurst=0.25,
        two_state_mode=True, p_fi=0.0, p_if=0.0, p_fns=0.0, p_ins=0.0,
        p_ns_out=0.0, p_dest_free=1.0, p_dest_inter=0.0,
        k_on=K_ON, k_off_site=np.full(len(SITES), 1.0 / TAU),
        enable_cooperativity=True,
        coop_strength=ON['strength'], coop_hill=ON['hill'],
        coop_off_strength=OFF['strength'], coop_off_hill=OFF['hill'],
        sample_ds=lambda st: np.full(len(st), 0.3))


def state(occupied_sites, probe_site=None):
    """Molecules on `occupied_sites`, plus (optionally) one free molecule
    10 nm from `probe_site`. Returns (SimState, probe molecule id)."""
    n = len(occupied_sites) + (probe_site is not None)
    pos = np.zeros((n, 3)); states = np.zeros(n, dtype=int)
    bsi = np.full(n, -1); occ = np.full(len(SITES), -1)
    for m, site in enumerate(occupied_sites):
        pos[m] = SITES[site]; states[m] = 2; bsi[m] = site; occ[site] = m
    probe = None
    if probe_site is not None:
        probe = n - 1
        pos[probe] = SITES[probe_site] + [0.01, 0.0, 0.0]
    counts = np.bincount(CLUSTER[occ >= 0], minlength=2)
    s = SimState(pos=pos, states=states, bound_site_indices=bsi,
                 current_ds=np.zeros(n), sites=SITES.copy(),
                 sites_base=SITES.copy(), site_occupancy=occ,
                 cluster_occupancy_count=counts, cached_tree=cKDTree(SITES))
    return s, probe


def check(label, hits, expected):
    p = hits / TRIALS
    sem = np.sqrt(expected * (1 - expected) / TRIALS)
    good = abs(p - expected) < 4 * sem
    print("%s %-34s p = %.4f  expected %.4f  (+- %.4f)"
          % ("ok  " if good else "FAIL", label, p, expected, sem))
    return good


def main():
    prm = params()
    rng = np.random.default_rng(0)
    ok = True

    print("binding: probe molecule next to free site 5, n sites of the cluster occupied")
    for n in range(0, 6):
        hits = 0
        for _ in range(TRIALS):
            s, probe = state(list(range(n)), probe_site=5)
            bind(s, prm, np.array([probe]), False, rng)
            hits += s.states[probe] == 2
        f = coop_factor([n], **ON)[0]
        ok &= check("n=%d  f_on=%.2f" % (n, f), hits, -np.expm1(-K_ON * f * DT))
    hits = 0
    for _ in range(TRIALS):
        s, probe = state(list(range(6)), probe_site=6)        # lone site
        bind(s, prm, np.array([probe]), False, rng)
        hits += s.states[probe] == 2
    ok &= check("lone site (cluster full)  f_on=1", hits, -np.expm1(-K_ON * DT))

    print("unbinding: molecule on site 0, n sites of the cluster occupied in total")
    for n in range(1, 7):
        hits = 0
        for _ in range(TRIALS):
            s, _ = state(list(range(n)))
            released = unbind(s, prm, rng)
            hits += 0 in released
        f = coop_factor([n - 1], **OFF)[0]
        ok &= check("n=%d  f_off=%.2f" % (n, f), hits, -np.expm1(-DT / TAU / f))
    hits = 0
    for _ in range(TRIALS):
        s, _ = state(list(range(6)) + [6])
        released = unbind(s, prm, rng)
        hits += 6 in released                                  # lone site molecule
    ok &= check("lone site (cluster full)  f_off=1", hits, -np.expm1(-DT / TAU))

    print("k_on_prob conversion")
    for prob in (0.05, 0.3, 0.9):
        k_on, src = binding_rate({'k_on_prob': prob, 'sim_dt': DT})
        back = -np.expm1(-k_on * DT)
        good = src == 'k_on_prob' and abs(back - prob) < 1e-12
        ok &= good
        print("%s k_on_prob %.2f -> k_on %.3f /s -> p %.12f" % (
            "ok  " if good else "FAIL", prob, k_on, back))
    good = binding_rate({'k_on_prob': 1.0, 'sim_dt': DT})[0] == np.inf
    ok &= good
    print("%s k_on_prob 1 -> k_on inf" % ("ok  " if good else "FAIL"))

    print("cooperativity probabilities: %s" % ("passed" if ok else "FAILED"))
    sys.exit(0 if ok else 1)


if __name__ == '__main__':
    main()
