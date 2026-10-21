"""
physics.py - One physics step of the simulation.

run_physics_step() applies, in this order:
    1. move_sites          cluster fBm drift + per-site jiggle
    2. diffuse             random walk of unbound molecules; bound ones
       enforce_boundary    follow their site; movers kept inside the nucleus
    3. unbind              first-order release of bound molecules
                           (ids returned)
    4. transitions         a) bind: molecules that diffused this step try
                              their nearest site
                           b) state_transitions: free <-> (intermediate)
                              <-> non-specific switching
A molecule changes state at most once per step: molecules released in 3
skip 4a and 4b (they can rebind from the next step on, after one
diffusion step), molecules that bound in 4a skip 4b, and a molecule that
switched in 4b does not switch again in the same step.

All arrays live in a SimState and are modified in place; the constants of
the run live in a StepParams. Site motion draws from site_rng, every
molecule event from mol_rng; the order of the draws is fixed by the order
of the stages above.

BINDING AND UNBINDING RATES
Both are first-order rates turned into per-step probabilities with the
exponential distribution, p = 1 - exp(-rate * dt):
    binding    rate = k_on * f_on(n)                (nearest site free and
                                                     within capture radius)
    unbinding  rate = k_off / f_off(n),  k_off = 1 / (res_time_mean * strength)
with the cooperativity factors
    f_on(n)  = 1 + (coop_strength * n) ** coop_hill
    f_off(n) = 1 + (coop_off_strength * n) ** coop_off_hill
where n is the number of OTHER molecules bound in the site's cluster at
the start of the stage (f = 1 for n = 0, so a lone site has no
cooperativity), and f = 1 when enable_cooperativity is off. k_on = inf
means binding on contact (p = 1).

STATE CODES: 0 free, 1 intermediate, 2 specifically bound, 3 non-specific.
"""

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np
from scipy.spatial import cKDTree

from .site_motion import generate_fbm_step


# ============================================================================
# STATE AND PARAMETERS
# ============================================================================

@dataclass
class SimState:
    """Everything that changes during the run (modified in place).

    Per molecule (N):  pos (N, 3) µm, states, bound_site_indices (-1 if not
                       bound), current_ds (µm²/s)
    Per site (S):      sites (S, 3) µm as imaged, sites_base (S, 3) µm
                       without jiggle, site_occupancy (molecule id or -1)
    Per cluster (K):   cluster_occupancy_count (occupied sites)
    Carried between steps: prev_cluster_step (last fBm step, (K, 3)) and
    cached_tree (KD-tree of the sites, reused while sites do not move).
    """
    pos: np.ndarray
    states: np.ndarray
    bound_site_indices: np.ndarray
    current_ds: np.ndarray
    sites: np.ndarray
    sites_base: np.ndarray
    site_occupancy: np.ndarray
    cluster_occupancy_count: np.ndarray
    prev_cluster_step: Optional[np.ndarray] = None
    cached_tree: Any = None


@dataclass(frozen=True)
class StepParams:
    """Constants of the run used by the physics step."""
    sim_dt_s: float
    # geometry (µm)
    radius_um: float            # nucleus radius
    inner_radius: float         # sites_base kept within this radius
    capture_radius_um: float
    # site landscape
    site_strengths: np.ndarray  # (S,) residence-time multiplier
    site_cluster_idx: np.ndarray  # (S,) cluster index 0..K-1
    cluster_D: np.ndarray       # (K,) µm²/s
    num_clusters: int
    # site motion
    bs_diff_mean: float
    bs_diff_std: float
    bs_jiggle_std: float
    bs_hurst: float
    # state switching (probabilities per step)
    two_state_mode: bool
    p_fi: float
    p_if: float
    p_fns: float
    p_ins: float
    p_ns_out: float
    p_dest_free: float          # fraction of leavers going to free ...
    p_dest_inter: float         # ... and to intermediate (three-state)
    # specific binding / unbinding (rates in 1/s, see module docstring)
    k_on: float                 # binding rate in capture range (inf allowed)
    k_off_site: np.ndarray      # (S,) 1 / (res_time_mean * strength)
    enable_cooperativity: bool
    coop_strength: float        # binding side
    coop_hill: float
    coop_off_strength: float    # unbinding side
    coop_off_hill: float
    # draws a diffusion coefficient for each state in an array
    sample_ds: Callable[[np.ndarray], np.ndarray]


# ============================================================================
# 1. SITE MOTION
# ============================================================================

def move_sites(s, prm, cluster_step, site_rng):
    """Move the binding sites.

    sites_base follows the cluster fBm (cluster_step precomputed for this
    frame, or a one-step fallback) and is kept within inner_radius; sites
    is sites_base plus an independent Gaussian jiggle, kept within the
    nucleus. Updates s.prev_cluster_step. Returns True if sites moved.
    """
    if not (prm.bs_diff_mean > 0 or prm.bs_diff_std > 0 or prm.bs_jiggle_std > 0):
        s.prev_cluster_step = None
        return False
    current_cluster_step = None
    if prm.bs_diff_mean > 0 or prm.bs_diff_std > 0:
        if cluster_step is None:
            cluster_step = generate_fbm_step(
                s.prev_cluster_step, prm.bs_hurst, prm.num_clusters,
                prm.cluster_D, prm.sim_dt_s, rng=site_rng)
        s.sites_base += cluster_step[prm.site_cluster_idx]
        current_cluster_step = cluster_step
    s_dists = np.linalg.norm(s.sites_base, axis=1)
    m_out = s_dists > prm.inner_radius
    s.sites_base[m_out] *= (prm.inner_radius / s_dists[m_out])[:, np.newaxis]
    if prm.bs_jiggle_std > 0:
        s.sites[:] = s.sites_base + site_rng.normal(0, prm.bs_jiggle_std, s.sites.shape)
        # Clamp sites to radius_um after jiggle
        _sd = np.linalg.norm(s.sites, axis=1)
        _so = _sd > prm.radius_um
        if np.any(_so):
            s.sites[_so] *= (prm.radius_um / _sd[_so])[:, np.newaxis]
    else:
        s.sites[:] = s.sites_base
    s.prev_cluster_step = current_cluster_step
    return True


# ============================================================================
# 2. DIFFUSION
# ============================================================================

def diffuse(s, prm, mol_rng):
    """Brownian step of every unbound molecule (std sqrt(2 D dt) per axis);
    bound molecules are placed on their site. Returns the indices of the
    molecules that moved (unbound at this point)."""
    mobile_idx = np.where(s.states != 2)[0]
    if len(mobile_idx) > 0:
        s.pos[mobile_idx] += mol_rng.normal(
            0, np.sqrt(2 * s.current_ds[mobile_idx] * prm.sim_dt_s)[:, np.newaxis],
            (len(mobile_idx), 3))
    bound_mask = (s.states == 2)
    if np.any(bound_mask):
        s.pos[bound_mask] = s.sites[s.bound_site_indices[bound_mask]]
    return mobile_idx


# ============================================================================
# 2. BOUNDARY (right after diffusion)
# ============================================================================

def enforce_boundary(s, prm, mobile_idx):
    """Pull molecules that moved outside the nucleus back onto its surface
    (bound molecules sit on sites, which are already inside)."""
    if len(mobile_idx) > 0:
        _bd = np.linalg.norm(s.pos[mobile_idx], axis=1)
        _bo = _bd > prm.radius_um
        if np.any(_bo):
            _oi = mobile_idx[_bo]
            s.pos[_oi] *= (prm.radius_um / _bd[_bo])[:, np.newaxis]


# ============================================================================
# 3. UNBINDING
# ============================================================================

_NONE = np.zeros(0, dtype=np.int64)


def coop_factor(n, strength, hill):
    """1 + (strength * n) ** hill, and exactly 1 where n == 0."""
    n = np.asarray(n, dtype=float)
    f = np.ones_like(n)
    m = n > 0
    f[m] = 1.0 + (strength * n[m]) ** hill
    return f


def rate_to_prob(rate, dt):
    """Per-step probability of a first-order event: 1 - exp(-rate * dt)
    (1 for an infinite rate)."""
    return -np.expm1(-np.asarray(rate, dtype=float) * dt)


def unbind(s, prm, mol_rng):
    """Release bound molecules: first-order unbinding.

    Each bound molecule leaves with p = 1 - exp(-k_off / f_off(n) * dt),
    k_off = 1 / (res_time_mean * site strength) and n the number of other
    molecules bound in its cluster at the start of this stage. Without
    cooperativity residence times are exponential with mean
    res_time_mean * strength. The site becomes free (and its cluster's
    occupancy count drops). The molecule stays where it is this step (on
    the site) and goes to free
    (two-state mode) or to free / intermediate in the ratio p_dest_free :
    p_dest_inter, with a new diffusion coefficient. Returns the ids of the
    released molecules.
    """
    bound_idx = np.where(s.states == 2)[0]
    if len(bound_idx) == 0:
        return _NONE
    bsite = s.bound_site_indices[bound_idx]
    rate = prm.k_off_site[bsite]
    if prm.enable_cooperativity and prm.coop_off_strength > 0:
        n_other = s.cluster_occupancy_count[prm.site_cluster_idx[bsite]] - 1
        rate = rate / coop_factor(n_other, prm.coop_off_strength, prm.coop_off_hill)
    p_off = rate_to_prob(rate, prm.sim_dt_s)
    rel_idx = bound_idx[mol_rng.random(len(bound_idx)) < p_off]
    if len(rel_idx) == 0:
        return _NONE
    target_sites = s.bound_site_indices[rel_idx]
    valid_releases = (s.site_occupancy[target_sites] == rel_idx)
    final_target_sites = target_sites[valid_releases]
    final_rel_idx = rel_idx[valid_releases]
    s.site_occupancy[final_target_sites] = -1
    np.subtract.at(s.cluster_occupancy_count,
                   prm.site_cluster_idx[final_target_sites], 1)
    if prm.two_state_mode:
        s.states[final_rel_idx] = mol_rng.choice(
            [0, 3], size=len(final_rel_idx), p=[prm.p_dest_free, 1.0 - prm.p_dest_free])
    else:
        s.states[final_rel_idx] = mol_rng.choice(
            [0, 1], size=len(final_rel_idx), p=[prm.p_dest_free, prm.p_dest_inter])
    s.bound_site_indices[final_rel_idx] = -1
    s.current_ds[final_rel_idx] = prm.sample_ds(s.states[final_rel_idx])
    return final_rel_idx


# ============================================================================
# 4a. BINDING
# ============================================================================

def bind(s, prm, mobile_idx, sites_moved, mol_rng):
    """Capture of molecules that diffused this step (mobile_idx) by free
    sites.

    Each of them looks only at its NEAREST site; if that site is free and
    within capture_radius_um, it binds with p = 1 - exp(-k_on * f_on(n) *
    dt), n = molecules bound in that site's cluster (all of them are
    "other", since the site itself is free). When several molecules pick the same
    site, a random one wins. A bound molecule sits on the site until it
    unbinds (see unbind). Updates s.cached_tree (rebuilt if sites moved).
    """
    if len(mobile_idx) == 0:
        return
    if s.cached_tree is None or sites_moved:
        s.cached_tree = cKDTree(s.sites)
    dists, s_ids = s.cached_tree.query(s.pos[mobile_idx], k=1,
                                       distance_upper_bound=prm.capture_radius_um)
    valid_check = (dists < prm.capture_radius_um) & (s_ids < len(s.sites))
    if not np.any(valid_check):
        return

    # Candidates: nearest site in range and free
    cand_mol = mobile_idx[valid_check]
    cand_site = s_ids[valid_check]
    site_free = (s.site_occupancy[cand_site] == -1)
    cand_mol = cand_mol[site_free]
    cand_site = cand_site[site_free]
    if len(cand_mol) == 0:
        return

    # Binding probability (with optional cooperativity)
    if prm.enable_cooperativity:
        n_occ = s.cluster_occupancy_count[prm.site_cluster_idx[cand_site]]
        rate = prm.k_on * coop_factor(n_occ, prm.coop_strength, prm.coop_hill)
    else:
        rate = prm.k_on
    probs = rate_to_prob(rate, prm.sim_dt_s)
    binds = mol_rng.random(len(cand_mol)) < probs
    cand_mol = cand_mol[binds]
    cand_site = cand_site[binds]
    if len(cand_mol) == 0:
        return

    # One random winner per site: shuffle, then keep the first per site
    shuffle = mol_rng.permutation(len(cand_mol))
    cand_mol = cand_mol[shuffle]
    cand_site = cand_site[shuffle]
    _, unique_site_idx = np.unique(cand_site, return_index=True)
    cand_mol = cand_mol[unique_site_idx]
    cand_site = cand_site[unique_site_idx]

    # Record the bindings
    s.states[cand_mol] = 2
    s.bound_site_indices[cand_mol] = cand_site
    s.pos[cand_mol] = s.sites[cand_site]
    s.site_occupancy[cand_site] = cand_mol
    s.current_ds[cand_mol] = 0.0
    np.add.at(s.cluster_occupancy_count, prm.site_cluster_idx[cand_site], 1)


# ============================================================================
# 4b. STATE TRANSITIONS (free / intermediate / non-specific)
# ============================================================================

def state_transitions(s, prm, mol_rng, skip_idx=None):
    """Switch unbound molecules between free / intermediate / non-specific.

    Each transition happens with a fixed probability per step (rate *
    sim_dt). Two-state mode: free <-> non-specific only. Three-state mode:
    free -> intermediate / non-specific, intermediate -> free /
    non-specific, non-specific -> free or intermediate (split
    p_dest_free : p_dest_inter). Molecules in skip_idx and molecules that
    already switched in this call do not switch. Molecules that switched
    draw a new diffusion coefficient.
    """
    states = s.states
    eligible = np.ones(len(states), dtype=bool)
    if skip_idx is not None:
        eligible[skip_idx] = False
    changed_indices = []

    def _done(idx):
        if len(idx) > 0:
            eligible[idx] = False
            changed_indices.append(idx)

    if prm.two_state_mode:
        idx = np.where((states == 0) & eligible)[0]
        if len(idx):
            r = mol_rng.random(size=len(idx))
            ns_changers = idx[r < prm.p_fns]
            states[ns_changers] = 3
            _done(ns_changers)
        idx = np.where((states == 3) & eligible)[0]
        if len(idx):
            r = mol_rng.random(size=len(idx))
            leavers = idx[r < prm.p_ns_out]
            states[leavers] = 0
            _done(leavers)
    else:
        idx = np.where((states == 0) & eligible)[0]
        if len(idx):
            r = mol_rng.random(size=len(idx))
            to_inter = idx[r < prm.p_fi]
            to_ns = idx[(r >= prm.p_fi) & (r < (prm.p_fi + prm.p_fns))]
            states[to_inter] = 1
            states[to_ns] = 3
            _done(np.concatenate([to_inter, to_ns]))
        idx = np.where((states == 1) & eligible)[0]
        if len(idx):
            r = mol_rng.random(size=len(idx))
            to_free = idx[r < prm.p_if]
            to_ns = idx[(r >= prm.p_if) & (r < (prm.p_if + prm.p_ins))]
            states[to_free] = 0
            states[to_ns] = 3
            _done(np.concatenate([to_free, to_ns]))
        idx = np.where((states == 3) & eligible)[0]
        if len(idx):
            r = mol_rng.random(size=len(idx))
            leavers = idx[r < prm.p_ns_out]
            states[leavers] = mol_rng.choice([0, 1], size=len(leavers),
                                             p=[prm.p_dest_free, prm.p_dest_inter])
            _done(leavers)

    if changed_indices:
        all_changed = np.concatenate(changed_indices)
        s.current_ds[all_changed] = prm.sample_ds(states[all_changed])


# ============================================================================
# FULL STEP
# ============================================================================

def transitions(s, prm, mobile_idx, released_idx, sites_moved, mol_rng,
                check_binding=True):
    """4a binding, then 4b state switching for everyone else.

    mobile_idx (molecules that diffused this step) never contains
    released_idx (those were still bound while diffusing), so released
    molecules cannot bind; they are skipped in the switching explicitly.
    Molecules that bind become state 2 and so are not switched either.
    """
    if check_binding:
        bind(s, prm, mobile_idx, sites_moved, mol_rng)
    state_transitions(s, prm, mol_rng, skip_idx=released_idx)


def run_physics_step(s, prm, site_rng, mol_rng, cluster_step=None,
                     check_binding=True, check_invariants=False):
    """Advance the simulation by one sim_dt (see the module docstring).

    cluster_step: this step's precomputed fBm cluster step (K, 3) or None.
    check_invariants: verify the consistency checks of assert_invariants()
    after the step (slow; for tests).
    """
    sites_moved = move_sites(s, prm, cluster_step, site_rng)
    mobile_idx = diffuse(s, prm, mol_rng)
    enforce_boundary(s, prm, mobile_idx)
    released_idx = unbind(s, prm, mol_rng)
    if check_invariants:
        released_states = s.states[released_idx].copy()
    transitions(s, prm, mobile_idx, released_idx, sites_moved, mol_rng,
                check_binding=check_binding)
    if check_invariants:
        assert_invariants(s, prm)
        assert np.array_equal(s.states[released_idx], released_states), \
            "a molecule released this step bound or switched in the same step"


def assert_invariants(s, prm):
    """Consistency of the state (raises AssertionError)."""
    bound = s.states == 2
    assert np.array_equal(bound, s.bound_site_indices >= 0), \
        "state 2 <-> bound_site_indices mismatch"
    ids = np.where(bound)[0]
    assert np.array_equal(s.site_occupancy[s.bound_site_indices[ids]], ids), \
        "site_occupancy does not point back to the bound molecule"
    assert np.sum(s.site_occupancy >= 0) == len(ids), \
        "occupied sites != bound molecules"
    occ = np.bincount(prm.site_cluster_idx[s.site_occupancy >= 0],
                      minlength=prm.num_clusters)
    assert np.array_equal(occ, s.cluster_occupancy_count), \
        "cluster_occupancy_count out of date"
    assert np.allclose(s.pos[ids], s.sites[s.bound_site_indices[ids]]), \
        "bound molecule not on its site"
    assert np.all(np.linalg.norm(s.pos, axis=1) <= prm.radius_um * (1 + 1e-9)), \
        "molecule outside the nucleus"
