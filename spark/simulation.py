"""
simulation.py - Full simulation: set-up and time loop.

run_full_simulation() builds the site landscape and the molecules, then runs
num_display_frames frames of steps_per_display physics steps each, recording
the camera exposure window at the start of every frame.
"""

import numpy as np
from scipy import stats
from scipy.spatial import cKDTree

from .sites import generate_clustered_sites, assign_site_strengths
from .site_motion import frame_fbm_steps
from .physics import SimState, StepParams, run_physics_step


def run_full_simulation(
    num_display_frames, num_binding_sites, nucleus_diam_nm, tf_concentration_nM,
    sim_dt_s, display_dt_s,
    pct_free_init, pct_inter_init, pct_ns_init,
    d_inter, d_free, d_ns,
    d_free_std, d_inter_std, d_ns_std,
    bs_diff_mean, bs_diff_std, bs_jiggle_std, bs_hurst,
    k_free_to_inter, k_inter_to_free,
    k_free_to_ns, k_inter_to_ns,
    ns_res_time,
    cluster_distribution,
    res_time_mean,
    capture_radius_nm, k_on,
    strength_multipliers=None, strength_percentages=None,
    cluster_correlated_strength=False,
    num_synthetic_sites=0, synthetic_strength=1.0,
    synthetic_cluster_radius_nm=150.0, cluster_radius_nm=150.0,
    exposure_time_s=0.05,
    enable_cooperativity=False, coop_strength=1.0, coop_hill=1.0,
    coop_off_strength=0.0, coop_off_hill=1.0,
    two_state_mode=True,
    binding_check_interval=1, progress_callback=None,
    # -----------------------------------------------------------------
    # Two independent numpy Generators (None -> fresh, unseeded):
    #   site_rng : site layout, strengths, cluster D, fBm motion, jiggle
    #   mol_rng  : molecule positions, transitions, diffusion, binding,
    #              residence times
    # Keeping them separate means the site landscape AND its motion are
    # identical for a given site seed, whatever the molecule parameters.
    # -----------------------------------------------------------------
    site_rng=None, mol_rng=None,
    # -----------------------------------------------------------------
    # Binding / single-molecule tracking hook (see the smt_tracking package).
    # Optional object with .start(**info) and .observe(step, pos, states,
    # sites, bound_site_indices). It is called at the same
    # pre-step snapshot point the exposure buffers use, so step k in the
    # tracker is exactly exposure sub-step k in the images. The tracker
    # only READS state and never touches an RNG, so passing one does not
    # change the simulation outcome for a given seed.
    # -----------------------------------------------------------------
    tracker=None,
    # Verify state consistency after every physics step (slow; tests).
    check_invariants=False,
):
    if site_rng is None:
        site_rng = np.random.default_rng()
    if mol_rng is None:
        mol_rng = np.random.default_rng()

    radius_um = (nucleus_diam_nm / 2) / 1000.0
    capture_radius_um = capture_radius_nm / 1000.0

    if sim_dt_s <= 0:
        sim_dt_s = 0.013
    steps_per_display = int(display_dt_s / sim_dt_s)
    if steps_per_display < 1:
        steps_per_display = 1

    # over how many time steps the position of the particle needs to be averaged
    steps_per_exposure = max(1, int(exposure_time_s / sim_dt_s))  

    vol_um3 = (4.0 / 3) * np.pi * (radius_um ** 3)
    num_molecules = int(((tf_concentration_nM * 1e-9) * (vol_um3 * 1e-15)) * 6.022e23)

    # Site landscape (positions, cluster membership, strengths)
    sites, cluster_ids, is_synthetic = generate_clustered_sites(
        num_binding_sites, radius_um, cluster_distribution,
        cluster_radius_nm=cluster_radius_nm,
        num_synthetic=num_synthetic_sites,
        synthetic_radius_nm=synthetic_cluster_radius_nm,
        rng=site_rng)
    if strength_multipliers is None:
        strength_multipliers = [1.0]
    if strength_percentages is None:
        strength_percentages = [100.0]
    site_strengths = assign_site_strengths(
        len(sites), cluster_ids, strength_multipliers, strength_percentages,
        is_synthetic, synthetic_strength, cluster_correlated_strength,
        rng=site_rng)

    site_occupancy = np.full(len(sites), -1, dtype=int)

    inner_radius = radius_um - capture_radius_um
    s_dists = np.linalg.norm(sites, axis=1)
    m_out = s_dists > inner_radius
    if np.any(m_out):
        sites[m_out] *= (inner_radius / s_dists[m_out])[:, np.newaxis]

    sites_base = sites.copy()

    unique_clusters = np.unique(cluster_ids)
    num_clusters = len(unique_clusters)
    cluster_D = np.maximum(
        0.005,
        site_rng.normal(bs_diff_mean, bs_diff_std, num_clusters))

    cluster_id_to_idx = {cid: i for i, cid in enumerate(unique_clusters)}
    site_cluster_idx = np.array([cluster_id_to_idx[cid] for cid in cluster_ids])
    cluster_occupancy_count = np.zeros(num_clusters, dtype=int)

    # random positioning of the molecules within the radius
    pos = mol_rng.normal(size=(num_molecules, 3))
    pos /= np.linalg.norm(pos, axis=1)[:, np.newaxis]
    pos *= (mol_rng.random(num_molecules) ** (1.0 / 3) * radius_um)[:, np.newaxis]

    # definition of the molecular states for the tfs [(diffusing, bound) or (diffusing, intermediate, bound)]
    if two_state_mode:
        total_pct = pct_free_init + pct_ns_init
        if total_pct == 0:
            p_dist = [1, 0]
        else:
            p_dist = [pct_free_init / total_pct, pct_ns_init / total_pct]
        states = mol_rng.choice([0, 3], size=num_molecules, p=p_dist)
        p_dest_free = 1.0
        p_dest_inter = 0.0
    else:
        total_pct = pct_free_init + pct_inter_init + pct_ns_init
        if total_pct == 0:
            p_dist = [1, 0, 0]
        else:
            p_dist = [pct_free_init / total_pct, pct_inter_init / total_pct,
                      pct_ns_init / total_pct]
        states = mol_rng.choice([0, 1, 3], size=num_molecules, p=p_dist)
        sum_mobile = pct_free_init + pct_inter_init
        if sum_mobile > 0:
            p_dest_free = pct_free_init / sum_mobile
            p_dest_inter = pct_inter_init / sum_mobile
        else:
            p_dest_free, p_dest_inter = 0.5, 0.5

    bound_site_indices = np.full(num_molecules, -1, dtype=int)
    p_fi = k_free_to_inter * sim_dt_s
    p_if = k_inter_to_free * sim_dt_s
    p_fns = k_free_to_ns * sim_dt_s
    p_ins = k_inter_to_ns * sim_dt_s
    p_ns_out = (1.0 / ns_res_time) * sim_dt_s if ns_res_time > 0 else 1.0
    # First-order unbinding rate of each site, 1 / (res_time_mean * strength)
    # (infinite, i.e. leave at once, if that is <= 0).
    mean_res = res_time_mean * np.asarray(site_strengths, dtype=float)
    k_off_site = np.full(len(mean_res), np.inf)
    k_off_site[mean_res > 0] = 1.0 / mean_res[mean_res > 0]

    def sample_ds(state_arr):
        "Seetting different diffusion coeficients for different molecules types"
        ds = np.zeros_like(state_arr, dtype=float)
        m0 = (state_arr == 0)
        ds[m0] = np.maximum(0, mol_rng.normal(d_free, d_free_std,
                                                size=np.sum(m0)))
        if not two_state_mode:
            m1 = (state_arr == 1)
            ds[m1] = np.maximum(0, mol_rng.normal(d_inter, d_inter_std,
                                                    size=np.sum(m1)))
        m_ns = (state_arr == 3)
        ds[m_ns] = np.maximum(0, mol_rng.normal(d_ns, d_ns_std,
                                                   size=np.sum(m_ns)))
        return ds

    current_ds = sample_ds(states)
    exposure_data = {f: None for f in range(num_display_frames)}
    bs_hurst = np.clip(bs_hurst, 0.01, 0.99)

    # Cluster fBm is generated per frame, conditioned on the previous step.
    sites_diffuse = (bs_diff_mean > 0 or bs_diff_std > 0)
    frame_cluster_steps = None

    physics_step = 0
    binding_check_interval = max(1, int(binding_check_interval))
    sites_are_static = (bs_diff_mean <= 0 and bs_diff_std <= 0
                        and bs_jiggle_std <= 0)

    # Everything the physics step changes, and the constants it uses.
    # The arrays are shared (not copied): pos, states, sites, ... below
    # always show the current state.
    sim = SimState(
        pos=pos, states=states, bound_site_indices=bound_site_indices,
        current_ds=current_ds,
        sites=sites, sites_base=sites_base, site_occupancy=site_occupancy,
        cluster_occupancy_count=cluster_occupancy_count,
        cached_tree=cKDTree(sites) if sites_are_static else None)
    prm = StepParams(
        sim_dt_s=sim_dt_s, radius_um=radius_um, inner_radius=inner_radius,
        capture_radius_um=capture_radius_um,
        site_strengths=site_strengths, site_cluster_idx=site_cluster_idx,
        cluster_D=cluster_D, num_clusters=num_clusters,
        bs_diff_mean=bs_diff_mean, bs_diff_std=bs_diff_std,
        bs_jiggle_std=bs_jiggle_std, bs_hurst=bs_hurst,
        two_state_mode=two_state_mode,
        p_fi=p_fi, p_if=p_if, p_fns=p_fns, p_ins=p_ins, p_ns_out=p_ns_out,
        p_dest_free=p_dest_free, p_dest_inter=p_dest_inter,
        k_on=k_on, k_off_site=k_off_site,
        enable_cooperativity=enable_cooperativity,
        coop_strength=coop_strength, coop_hill=coop_hill,
        coop_off_strength=coop_off_strength, coop_off_hill=coop_off_hill,
        sample_ds=sample_ds)

    if tracker is not None:
        tracker.start(
            num_molecules=num_molecules,
            is_synthetic=np.asarray(is_synthetic, dtype=bool),
            site_strengths=np.asarray(site_strengths, dtype=float),
            sim_dt_s=sim_dt_s,
            steps_per_display=steps_per_display,
            steps_per_exposure=steps_per_exposure,
            num_display_frames=num_display_frames,
        )

    for display_frame in range(num_display_frames):
        # Generate fBm steps for this frame only
        if sites_diffuse:
            frame_cluster_steps = frame_fbm_steps(
                steps_per_display + 2, num_clusters, cluster_D, bs_hurst,
                sim_dt_s, prev_step=sim.prev_cluster_step, rng=site_rng)

        # ---------------------------------------------------------------
        # Unified per-display-frame loop. Each frame runs EXACTLY
        # steps_per_display physics steps. The exposure window sits at
        # the START of the frame: the first steps_per_exposure steps
        # snapshot into exposure_data[display_frame]. The remaining
        # (steps_per_display - steps_per_exposure) steps just advance
        # physics.
        # ---------------------------------------------------------------
        _exp_pos_buf    = []
        _exp_states_buf = []
        _exp_sites_buf  = []
        _exp_occ_buf    = []

        for frame_step in range(steps_per_display):
            in_exposure_window = (frame_step < steps_per_exposure)

            if in_exposure_window:
                # float32 halves size vs float64; astype() makes a copy
                _exp_pos_buf.append(pos.astype(np.float32))
                _exp_states_buf.append(states.copy())
                _exp_sites_buf.append(sites.astype(np.float32))
                _exp_occ_buf.append((site_occupancy >= 0).copy())

            if tracker is not None:
                tracker.observe(physics_step, pos, states, sites,
                                bound_site_indices)

            # Binding check policy: always check inside the exposure
            # window (matches the old exposure-loop behavior); honor
            # binding_check_interval outside it (matches the old
            # advance-loop behavior).
            check_binding = (in_exposure_window
                             or (physics_step % binding_check_interval == 0))

            cluster_step = (frame_cluster_steps[frame_step]
                            if frame_cluster_steps is not None else None)
            run_physics_step(sim, prm, site_rng, mol_rng,
                             cluster_step=cluster_step,
                             check_binding=check_binding,
                             check_invariants=check_invariants)
            physics_step += 1

        if _exp_pos_buf:
            _ps = np.stack(_exp_pos_buf)    # (steps, N, 3) float32
            _ss = np.stack(_exp_states_buf)
            _si = np.stack(_exp_sites_buf)
            _oo = np.stack(_exp_occ_buf)
            exposure_data[display_frame] = {
                'pos_steps':     _ps.astype(np.float32),               # (steps, N, 3)
                'modal_states':  stats.mode(_ss, axis=0, keepdims=False)[0].astype(np.int8),
                'site_centroid': _si.mean(axis=0).astype(np.float32),
                'any_occupied':  _oo.any(axis=0),
            }

        if progress_callback:
            progress_callback(display_frame + 1, num_display_frames)

    # Final observation captures any binding change during the last step.
    if tracker is not None:
        tracker.observe(physics_step, pos, states, sites,
                        bound_site_indices)

    return exposure_data, num_molecules, site_strengths, is_synthetic, \
        steps_per_exposure
