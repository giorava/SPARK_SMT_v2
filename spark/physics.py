"""
physics.py - One physics step of the simulation.

run_physics_step() applies, in this order:
    1. state_transitions   free <-> intermediate <-> non-specific switching
    2. move_sites          cluster fBm drift + per-site jiggle
    3. unbind              release molecules whose residence time is over
    4. diffuse             random walk of unbound molecules; bound ones
                           follow their site
    5. bind                capture by free sites within the capture radius
    6. enforce_boundary    keep moving molecules inside the nucleus
All arrays are modified in place. Site motion draws from site_rng, every
molecule event from mol_rng; the order of the draws is fixed by the order
of the stages above.

STATE CODES: 0 free, 1 intermediate, 2 specifically bound, 3 non-specific.
"""

import numpy as np
from scipy.spatial import cKDTree

from .site_motion import generate_fbm_step


# ============================================================================
# 1. STATE TRANSITIONS
# ============================================================================

def state_transitions(states, current_ds, sample_ds,
                      p_fi, p_if, p_fns, p_ins, p_ns_out,
                      p_dest_free, p_dest_inter, two_state_mode, mol_rng):
    """Switch unbound molecules between free / intermediate / non-specific.

    Each transition happens with a fixed probability per step (rate *
    sim_dt). Two-state mode: free <-> non-specific only. Three-state mode:
    free -> intermediate / non-specific, intermediate -> free /
    non-specific, non-specific -> free or intermediate (split
    p_dest_free : p_dest_inter). Molecules that changed state draw a new
    diffusion coefficient.
    """
    changed_indices = []
    if two_state_mode:
        mask_0 = (states == 0)
        if np.any(mask_0):
            r = mol_rng.random(size=np.sum(mask_0))
            to_ns = r < p_fns
            idx = np.where(mask_0)[0]
            ns_changers = idx[to_ns]
            states[ns_changers] = 3
            if len(ns_changers) > 0:
                changed_indices.append(ns_changers)
        mask_3 = (states == 3)
        if np.any(mask_3):
            r = mol_rng.random(size=np.sum(mask_3))
            leaving = r < p_ns_out
            idx = np.where(mask_3)[0]
            leavers = idx[leaving]
            states[leavers] = 0
            if len(leavers) > 0:
                changed_indices.append(leavers)
    else:
        mask_0 = (states == 0)
        if np.any(mask_0):
            r = mol_rng.random(size=np.sum(mask_0))
            to_inter = r < p_fi
            to_ns = (r >= p_fi) & (r < (p_fi + p_fns))
            idx = np.where(mask_0)[0]
            changed_0 = np.concatenate([idx[to_inter], idx[to_ns]])
            states[idx[to_inter]] = 1
            states[idx[to_ns]] = 3
            if len(changed_0) > 0:
                changed_indices.append(changed_0)
        mask_1 = (states == 1)
        if np.any(mask_1):
            r = mol_rng.random(size=np.sum(mask_1))
            to_free = r < p_if
            to_ns = (r >= p_if) & (r < (p_if + p_ins))
            idx = np.where(mask_1)[0]
            changed_1 = np.concatenate([idx[to_free], idx[to_ns]])
            states[idx[to_free]] = 0
            states[idx[to_ns]] = 3
            if len(changed_1) > 0:
                changed_indices.append(changed_1)
        mask_3 = (states == 3)
        if np.any(mask_3):
            r = mol_rng.random(size=np.sum(mask_3))
            leaving = r < p_ns_out
            idx = np.where(mask_3)[0]
            leavers = idx[leaving]
            new_dest = mol_rng.choice([0, 1], size=len(leavers), p=[p_dest_free, p_dest_inter])
            states[leavers] = new_dest
            if len(leavers) > 0:
                changed_indices.append(leavers)

    if changed_indices:
        all_changed = np.concatenate(changed_indices)
        current_ds[all_changed] = sample_ds(states[all_changed])


# ============================================================================
# 2. SITE MOTION
# ============================================================================

def move_sites(sites, sites_base, cluster_D, site_cluster_idx, num_clusters,
               bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
               radius_um, inner_radius, sim_dt_s,
               prev_cluster_step, precomputed_cluster_step, site_rng):
    """Move the binding sites.

    sites_base follows the cluster fBm (precomputed step for this frame,
    or a one-step fallback) and is kept within inner_radius; sites is
    sites_base plus an independent Gaussian jiggle, kept within the
    nucleus. Returns (sites_moved, cluster_step used or None).
    """
    if not (bs_diff_mean > 0 or bs_diff_std > 0 or bs_jiggle_std > 0):
        return False, None
    current_cluster_step = None
    if bs_diff_mean > 0 or bs_diff_std > 0:
        if precomputed_cluster_step is not None:
            cluster_step = precomputed_cluster_step
        else:
            cluster_step = generate_fbm_step(
                prev_cluster_step, bs_hurst, num_clusters, cluster_D, sim_dt_s,
                rng=site_rng)
        sites_base += cluster_step[site_cluster_idx]
        current_cluster_step = cluster_step
    s_dists = np.linalg.norm(sites_base, axis=1)
    m_out = s_dists > inner_radius
    sites_base[m_out] *= (inner_radius / s_dists[m_out])[:, np.newaxis]
    if bs_jiggle_std > 0:
        sites[:] = sites_base + site_rng.normal(0, bs_jiggle_std, sites.shape)
        # Clamp sites to radius_um after jiggle
        _sd = np.linalg.norm(sites, axis=1)
        _so = _sd > radius_um
        if np.any(_so):
            sites[_so] *= (radius_um / _sd[_so])[:, np.newaxis]
    else:
        sites[:] = sites_base
    return True, current_cluster_step


# ============================================================================
# 3. UNBINDING
# ============================================================================

def unbind(states, site_occupancy, bound_site_indices, unbind_times,
           current_ds, sample_ds, current_time,
           site_cluster_idx, enable_cooperativity, cluster_occupancy_count,
           p_dest_free, p_dest_inter, two_state_mode, mol_rng):
    """Release bound molecules whose unbind time has passed.

    The site becomes free (and its cluster's occupancy count drops). The
    molecule goes to free (two-state mode) or to free / intermediate in
    the ratio p_dest_free : p_dest_inter, with a new diffusion coefficient.
    """
    mask_bound = (states == 2)
    if not np.any(mask_bound):
        return
    releasing = mask_bound & (unbind_times <= current_time)
    if not np.any(releasing):
        return
    rel_idx = np.where(releasing)[0]
    target_sites = bound_site_indices[rel_idx]
    valid_releases = (site_occupancy[target_sites] == rel_idx)
    final_target_sites = target_sites[valid_releases]
    final_rel_idx = rel_idx[valid_releases]
    site_occupancy[final_target_sites] = -1
    if enable_cooperativity:
        c_indices = site_cluster_idx[final_target_sites]
        np.subtract.at(cluster_occupancy_count, c_indices, 1)
        cluster_occupancy_count[cluster_occupancy_count < 0] = 0
    if two_state_mode:
        states[final_rel_idx] = mol_rng.choice(
            [0, 3], size=len(final_rel_idx), p=[p_dest_free, 1.0 - p_dest_free])
    else:
        states[final_rel_idx] = mol_rng.choice(
            [0, 1], size=len(final_rel_idx), p=[p_dest_free, p_dest_inter])
    bound_site_indices[final_rel_idx] = -1
    current_ds[final_rel_idx] = sample_ds(states[final_rel_idx])


# ============================================================================
# 4. DIFFUSION
# ============================================================================

def diffuse(pos, states, current_ds, sites, bound_site_indices, sim_dt_s, mol_rng):
    """Brownian step of every unbound molecule (std sqrt(2 D dt) per axis);
    bound molecules are placed on their site. Returns the indices of the
    molecules that moved (unbound at this point)."""
    mobile_idx = np.where(states != 2)[0]
    if len(mobile_idx) > 0:
        pos[mobile_idx] += mol_rng.normal(
            0, np.sqrt(2 * current_ds[mobile_idx] * sim_dt_s)[:, np.newaxis], (len(mobile_idx), 3))
    bound_mask = (states == 2)
    if np.any(bound_mask):
        pos[bound_mask] = sites[bound_site_indices[bound_mask]]
    return mobile_idx


# ============================================================================
# 5. BINDING
# ============================================================================

def bind(pos, states, sites, site_occupancy, bound_site_indices, unbind_times,
         current_ds, mobile_idx, site_strengths, site_cluster_idx,
         capture_radius_um, k_on_prob, res_time_mean, res_time_std, current_time,
         enable_cooperativity, coop_strength, coop_hill, cluster_occupancy_count,
         cached_tree, sites_moved, mol_rng):
    """Capture of moving molecules by free sites.

    Each moving molecule looks only at its NEAREST site; if that site is
    free and within capture_radius_um, it binds with probability
    k_on_prob (times 1 + (coop_strength * n_occupied_in_cluster)^coop_hill
    with cooperativity, capped at 1). When several molecules pick the same
    site, a random one wins. A bound molecule sits on the site and gets an
    unbind time current_time + max(0.1, N(res_time_mean, res_time_std) *
    site strength). Returns the site KD-tree (rebuilt if sites moved).
    """
    new_tree = cached_tree
    if len(mobile_idx) == 0:
        return new_tree
    if cached_tree is None or sites_moved:
        new_tree = cKDTree(sites)
    dists, s_ids = new_tree.query(pos[mobile_idx], k=1, distance_upper_bound=capture_radius_um)
    valid_check = (dists < capture_radius_um) & (s_ids < len(sites))
    if not np.any(valid_check):
        return new_tree

    # Candidates: nearest site in range and free
    cand_mol = mobile_idx[valid_check]
    cand_site = s_ids[valid_check]
    site_free = (site_occupancy[cand_site] == -1)
    cand_mol = cand_mol[site_free]
    cand_site = cand_site[site_free]
    if len(cand_mol) == 0:
        return new_tree

    # Binding probability (with optional cooperativity)
    if enable_cooperativity:
        c_ids = site_cluster_idx[cand_site]
        n_occ = cluster_occupancy_count[c_ids]
        mult = 1.0 + (coop_strength * n_occ) ** coop_hill
        probs = np.minimum(1.0, k_on_prob * mult)
    else:
        probs = k_on_prob
    binds = mol_rng.random(len(cand_mol)) < probs
    cand_mol = cand_mol[binds]
    cand_site = cand_site[binds]
    if len(cand_mol) == 0:
        return new_tree

    # One random winner per site: shuffle, then keep the first per site
    shuffle = mol_rng.permutation(len(cand_mol))
    cand_mol = cand_mol[shuffle]
    cand_site = cand_site[shuffle]
    _, unique_site_idx = np.unique(cand_site, return_index=True)
    cand_mol = cand_mol[unique_site_idx]
    cand_site = cand_site[unique_site_idx]

    # Record the bindings
    states[cand_mol] = 2
    bound_site_indices[cand_mol] = cand_site
    pos[cand_mol] = sites[cand_site]
    site_occupancy[cand_site] = cand_mol
    current_ds[cand_mol] = 0.0
    if enable_cooperativity:
        np.add.at(cluster_occupancy_count, site_cluster_idx[cand_site], 1)
    site_str = site_strengths[cand_site]
    unbind_times[cand_mol] = current_time + np.maximum(
        0.1, mol_rng.normal(res_time_mean * site_str, res_time_std * site_str))
    return new_tree


# ============================================================================
# 6. BOUNDARY
# ============================================================================

def enforce_boundary(pos, mobile_idx, radius_um):
    """Pull molecules that moved outside the nucleus back onto its surface
    (bound molecules sit on sites, which are already inside)."""
    if len(mobile_idx) > 0:
        _bd = np.linalg.norm(pos[mobile_idx], axis=1)
        _bo = _bd > radius_um
        if np.any(_bo):
            _oi = mobile_idx[_bo]
            pos[_oi] *= (radius_um / _bd[_bo])[:, np.newaxis]


# ============================================================================
# FULL STEP
# ============================================================================

def run_physics_step(
    pos, states, sites, sites_base, site_occupancy, bound_site_indices, unbind_times,
    current_ds, sample_ds, cluster_D, site_cluster_idx, site_strengths,
    p_fi, p_if, p_fns, p_ins, p_ns_out, p_dest_free, p_dest_inter,
    bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
    radius_um, capture_radius_um, inner_radius, sim_dt_s, current_time,
    res_time_mean, res_time_std, k_on_prob, num_clusters,
    prev_cluster_step,
    enable_cooperativity, coop_strength, coop_hill, cluster_occupancy_count,
    two_state_mode=False, check_binding=True, cached_tree=None,
    precomputed_cluster_step=None,
    site_rng=None, mol_rng=None,
):
    """Advance the simulation by one sim_dt (see the module docstring).

    Returns (cluster step used this step, site KD-tree) for the next call.
    """
    # Site motion (fBm fallback, jiggle) uses site_rng; every molecule
    # event uses mol_rng.
    if site_rng is None:
        site_rng = np.random
    if mol_rng is None:
        mol_rng = np.random

    state_transitions(states, current_ds, sample_ds,
                      p_fi, p_if, p_fns, p_ins, p_ns_out,
                      p_dest_free, p_dest_inter, two_state_mode, mol_rng)

    sites_moved, current_cluster_step = move_sites(
        sites, sites_base, cluster_D, site_cluster_idx, num_clusters,
        bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
        radius_um, inner_radius, sim_dt_s,
        prev_cluster_step, precomputed_cluster_step, site_rng)

    unbind(states, site_occupancy, bound_site_indices, unbind_times,
           current_ds, sample_ds, current_time,
           site_cluster_idx, enable_cooperativity, cluster_occupancy_count,
           p_dest_free, p_dest_inter, two_state_mode, mol_rng)

    mobile_idx = diffuse(pos, states, current_ds, sites, bound_site_indices,
                         sim_dt_s, mol_rng)

    new_tree = cached_tree
    if check_binding:
        new_tree = bind(pos, states, sites, site_occupancy, bound_site_indices,
                        unbind_times, current_ds, mobile_idx, site_strengths,
                        site_cluster_idx, capture_radius_um, k_on_prob,
                        res_time_mean, res_time_std, current_time,
                        enable_cooperativity, coop_strength, coop_hill,
                        cluster_occupancy_count, cached_tree, sites_moved, mol_rng)

    enforce_boundary(pos, mobile_idx, radius_um)
    return current_cluster_step, new_tree
