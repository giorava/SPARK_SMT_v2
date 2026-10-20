"""
sites.py - Binding-site landscape: positions, cluster membership, strengths.

All draws use the generator passed as `rng` (the simulation's site stream).
"""

import numpy as np


def generate_clustered_sites(num_sites, nucleus_radius_um, cluster_probs,
                             cluster_radius_nm=150.0, num_synthetic=0,
                             synthetic_radius_nm=150.0, rng=None):
    if rng is None:
        rng = np.random
    cluster_radius_um = cluster_radius_nm / 1000.0
    synthetic_radius_um = synthetic_radius_nm / 1000.0
    sites, cluster_ids, is_synthetic = [], [], []
    cid = 0

    total_p = sum(cluster_probs.values())
    if total_p == 0 and num_sites > 0:
        norm = {1: 1.0}
    elif total_p > 0:
        norm = {k: v / total_p for k, v in cluster_probs.items()}
    else:
        norm = {}

    for size, prob in norm.items():
        n_clusters = int(int(num_sites * prob) / size)
        for _ in range(n_clusters):
            center = rng.normal(size=3)
            center /= np.linalg.norm(center)
            center *= (rng.random() ** (1.0 / 3)) * nucleus_radius_um
            for _ in range(size):
                off = rng.normal(size=3)
                off /= np.linalg.norm(off)
                off *= (rng.random() ** (1.0 / 3)) * cluster_radius_um
                pos = center + off
                if np.linalg.norm(pos) > nucleus_radius_um:
                    pos = pos / np.linalg.norm(pos) * nucleus_radius_um
                sites.append(pos)
                cluster_ids.append(cid)
                is_synthetic.append(False)
            cid += 1

    while len(sites) < num_sites:
        pos = rng.normal(size=3)
        pos /= np.linalg.norm(pos)
        pos *= (rng.random() ** (1.0 / 3)) * nucleus_radius_um
        sites.append(pos)
        cluster_ids.append(cid)
        is_synthetic.append(False)
        cid += 1

    if num_synthetic > 0:
        sc = rng.normal(size=3)
        sc /= np.linalg.norm(sc)
        sc *= (rng.random() ** (1.0 / 3)) * nucleus_radius_um * 0.7
        syn_cid = cid
        for _ in range(num_synthetic):
            off = rng.normal(size=3)
            off /= np.linalg.norm(off)
            off *= (rng.random() ** (1.0 / 3)) * synthetic_radius_um
            pos = sc + off
            if np.linalg.norm(pos) > nucleus_radius_um:
                pos = pos / np.linalg.norm(pos) * nucleus_radius_um
            sites.append(pos)
            cluster_ids.append(syn_cid)
            is_synthetic.append(True)

    return np.array(sites), np.array(cluster_ids), np.array(is_synthetic)


def assign_site_strengths(num_sites, cluster_ids, strength_multipliers,
                          strength_percentages, is_synthetic,
                          synthetic_strength=1.0, cluster_correlated=False,
                          rng=None):
    if rng is None:
        rng = np.random
    site_strengths = np.zeros(num_sites)
    # synthetic_strength may be scalar (broadcast to every synthetic site) or
    # a 1-D array of length equal to the number of synthetic sites (per-site
    # multiplier; element i -> i-th synthetic site in generation order).
    num_synth = int(np.sum(is_synthetic))
    if num_synth > 0:
        ss_arr = np.atleast_1d(np.asarray(synthetic_strength, dtype=float))
        if ss_arr.size not in (1, num_synth):
            raise ValueError(
                "synthetic_strength has length %d but there are %d synthetic "
                "sites; pass either a scalar or an array of length %d."
                % (ss_arr.size, num_synth, num_synth)
            )
        site_strengths[is_synthetic] = ss_arr  # broadcasts if size 1
    endogenous_mask = ~is_synthetic
    num_endogenous = np.sum(endogenous_mask)
    if num_endogenous > 0:
        total_pct = sum(strength_percentages)
        if total_pct == 0:
            probs = [1.0 / len(strength_multipliers)] * len(strength_multipliers)
        else:
            probs = [p / total_pct for p in strength_percentages]
        if cluster_correlated:
            endogenous_clusters = np.unique(cluster_ids[endogenous_mask])
            cluster_strengths_dict = {}
            for c in endogenous_clusters:
                cluster_strengths_dict[c] = rng.choice(strength_multipliers, p=probs)
            for idx in np.where(endogenous_mask)[0]:
                site_strengths[idx] = cluster_strengths_dict[cluster_ids[idx]]
        else:
            eidx = np.where(endogenous_mask)[0]
            site_strengths[eidx] = rng.choice(
                strength_multipliers, size=num_endogenous, p=probs)
    return site_strengths
