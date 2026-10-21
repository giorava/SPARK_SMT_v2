"""
pipeline.py - Everything done for one CSV row.

process_row()            simulate -> tracking / SMT -> ground truth ->
                         convolved images (AU now, or grids kept for
                         auto-normalize) -> tables -> hyperstacks
render_normalized_row()  auto-normalize phase 2: render the kept grids
                         with the vmax shared by all rows

Both take (row_idx, row_dict, config) so they can be used with Pool.map.
`config` is the dict built from the command line in batch_hpc_fast.py.
"""

import os
import time

import numpy as np
import pandas as pd

import smt_tracking

from .params import extract_row_params
from .seeds import row_rngs, render_rng_from_entropy
from .simulation import run_full_simulation
from .exposure import compute_exposure_frames
from .convolution import (exposure_psf_grid, render_au_slices,
                          render_normalized_slices)
from .ground_truth import render_ground_truth
from .log import log
from . import outputs


# ============================================================================
# SIMULATION
# ============================================================================

def simulate_row(row, p, rngs, tracker=None, progress_callback=None):
    """run_full_simulation() with the parameters of one CSV row.

    Returns (exposure_data, n_molecules, site_strengths, is_synthetic,
    steps_per_exposure).
    """
    f = lambda key: float(row[key])
    return run_full_simulation(
        num_display_frames=int(row['num_frames']),
        num_binding_sites=int(row['num_binding_sites']),
        nucleus_diam_nm=f('nucleus_diameter_nm'),
        tf_concentration_nM=f('tf_concentration_nM'),
        sim_dt_s=f('sim_dt'), display_dt_s=f('display_dt'),
        exposure_time_s=p['exposure_time_s'],
        # initial state fractions
        pct_free_init=f('pct_free_init'), pct_inter_init=f('pct_inter_init'),
        pct_ns_init=f('pct_ns_init'),
        # TF diffusion coefficients (µm²/s)
        d_free=f('d_free'), d_free_std=f('d_free_std'),
        d_inter=f('d_inter'), d_inter_std=f('d_inter_std'),
        d_ns=f('d_ns'), d_ns_std=f('d_ns_std'),
        # state switching
        two_state_mode=p['two_state_mode'],
        k_free_to_inter=f('k_fi'), k_inter_to_free=f('k_if'),
        k_free_to_ns=f('k_fns'), k_inter_to_ns=f('k_ins'),
        ns_res_time=f('ns_res'),
        # specific binding
        capture_radius_nm=f('capture_rad_nm'), k_on=p['k_on'],
        res_time_mean=f('res_time_mean'),
        enable_cooperativity=p['enable_coop'],
        coop_strength=p['coop_strength'], coop_hill=p['coop_hill'],
        coop_off_strength=p['coop_off_strength'], coop_off_hill=p['coop_off_hill'],
        # binding sites
        cluster_distribution=p['cluster_dist'],
        cluster_radius_nm=p['cluster_radius_nm'],
        strength_multipliers=p['strength_multipliers'],
        strength_percentages=p['strength_percentages'],
        cluster_correlated_strength=p['cluster_correlated'],
        num_synthetic_sites=p['num_synthetic'],
        synthetic_strength=p['synthetic_strength'],
        synthetic_cluster_radius_nm=p['synthetic_radius'],
        # site motion
        bs_diff_mean=f('bs_diff_mean'), bs_diff_std=f('bs_diff_std'),
        bs_jiggle_std=f('bs_jiggle_std'), bs_hurst=float(p['bs_hurst']),
        # randomness / hooks
        site_rng=rngs.site, mol_rng=rngs.mol,
        tracker=tracker, progress_callback=progress_callback,
    )


def _label_spec(row, config):
    """A non-empty smt_label cell in the CSV overrides --smt-label."""
    spec = row.get('smt_label', None)
    if isinstance(spec, str) and spec.strip():
        return spec.strip()
    return config.get('smt_label', 'random:1')


def _au_calibration(row, p):
    return {
        'ref_concentration': p['au_ref_concentration_nM'],
        'ref_signal': p['au_ref_signal'],
        'background': p['au_background'],
        'noise_std': p['au_noise_std'],
        'sim_concentration': float(row['tf_concentration_nM']),
    }


# ============================================================================
# PHASE 1 / SINGLE PASS
# ============================================================================

def process_row(args_tuple):
    """Simulate one CSV row and write its outputs.

    Convolved images are rendered here in absolute units; with
    auto-normalize the 3D grids and the row maximum are kept for
    render_normalized_row() instead. Returns (row_idx, row_folder, row_max).
    """
    row_idx, row_dict, config = args_tuple
    row = pd.Series(row_dict)
    p = extract_row_params(row)
    nuc_nm = float(row['nucleus_diameter_nm'])
    voxel_nm = float(row['voxel_size_nm'])
    slice_nm = float(row['slice_spacing_nm'])
    bit_depth = config['bit_depth']
    tag = "[Row %d]" % row_idx

    rngs = row_rngs(config['seed'], row_idx, config.get('shared_sites'))
    folder = outputs.row_folder(row_idx, p)
    row_outdir = os.path.join(config['outdir'], folder)
    os.makedirs(row_outdir, exist_ok=True)
    outputs.write_run_info(row_outdir, row_idx, config['seed'],
                           rngs.seed_sequence, config.get('shared_sites'))

    # ---- Simulation -------------------------------------------------------
    t0 = time.time()
    log("%s Simulating: %s  (n_frames=%s, n_sites=%s, conc=%s nM)" % (
        tag, p['row_label'], row['num_frames'], row['num_binding_sites'],
        row['tf_concentration_nM']))

    def progress(done, total):
        if done % max(1, total // 5) == 0 or done == total:
            log("  %s Frame %d/%d" % (tag, done, total))

    rts = row.get('res_time_std', None)
    if rts is not None and pd.notna(rts) and float(rts) != 0:
        log("%s Note: res_time_std (%g) is ignored; residence times are "
            "exponential with mean res_time_mean * site strength" % (tag, float(rts)))

    if p['k_on_source'] == 'k_on_prob':
        log("%s Note: k_on_prob %g per step -> binding rate k_on = %g /s "
            "(add a k_on column to set the rate directly)"
            % (tag, float(row['k_on_prob']), p['k_on']))

    label_spec = _label_spec(row, config)
    tracker = None
    if config.get('record_binding') or config.get('smt'):
        # Read-only observer: does not consume random numbers.
        tracker = smt_tracking.BindingTracker(
            label_spec=label_spec,
            trajectory_stride=config.get('trajectory_stride', 0),
            label_seed=rngs.label_seed)

    exposure_data, n_mol, site_strengths, is_synthetic, spe = simulate_row(
        row, p, rngs, tracker=tracker, progress_callback=progress)
    log("%s Simulation complete: %d molecules" % (tag, n_mol))

    mol_data, site_data = compute_exposure_frames(exposure_data)
    frames = sorted(mol_data.keys())

    # ---- Binding tables, labelled tracks, SMT images ------------------------
    if tracker is not None:
        smt_tracking.write_tracking_outputs(
            row_outdir, tracker.result(), exposure_data, label_spec,
            smt_tracking.make_label_rng(rngs.label_seed),
            smt_tracking.image_geometry(nuc_nm, voxel_nm, slice_nm, p['num_slices']),
            num_frames=int(row['num_frames']),
            spd=max(1, int(float(row['display_dt']) / float(row['sim_dt']))),
            spe=int(spe),
            lateral_fwhm_nm=p['lateral_fwhm'], axial_fwhm_nm=p['axial_fwhm'],
            near_radius_nm=config.get('near_radius_nm', 250.0),
            write_smt=bool(config.get('smt')),
            smt_params=config.get('smt_params'),
            bit_depth=bit_depth,
            log=lambda m: log(tag + m))

    # ---- Ground truth -------------------------------------------------------
    if config['save_gt']:
        for frame in frames:
            sites_pos, site_occ = site_data.get(frame, (None, None))
            images = render_ground_truth(
                mol_data[frame], nuc_nm, float(row['molecule_diameter_nm']),
                int(row['image_dpi']), slice_nm, sites_pos, site_occ,
                site_strengths=site_strengths, strength_range=p['strength_range'],
                is_synthetic=is_synthetic, upscale_factor=p['upscale_factor'],
                voxel_size_nm=voxel_nm, num_slices=p['num_slices'])
            outputs.write_frame_stack(row_outdir, 'Ground_Truth', frame, images)

    # ---- All-molecule convolved images ----------------------------------------
    row_max = 0.0
    if config['save_conv']:
        grids = {}
        for frame in frames:
            grid = exposure_psf_grid(exposure_data, frame, nuc_nm, voxel_nm,
                                     p['lateral_fwhm'], p['axial_fwhm'])
            if grid is not None:
                grids[frame] = grid
                row_max = max(row_max, float(grid.max()))
        log("%s Grids computed: %d grids, max_intensity=%.6f"
            % (tag, len(grids), row_max))

        if config['auto_normalize']:
            # Rendering waits for the maximum over all rows (phase 2).
            np.savez_compressed(os.path.join(row_outdir, '_grids.npz'),
                                **{'f%d' % fr: g for fr, g in grids.items()})
            outputs.write_row_max(row_outdir, row_max)
        else:
            au_cal = _au_calibration(row, p)
            for frame in sorted(grids):
                images = render_au_slices(
                    grids[frame], nuc_nm, voxel_nm, slice_nm, au_cal,
                    num_slices=p['num_slices'], bit_depth=bit_depth,
                    upscale_factor=config['upscale_factor'], rng=rngs.render)
                outputs.write_frame_stack(
                    row_outdir, 'Convolved_AU_%dbit' % bit_depth, frame, images)

    # ---- Tables and hyperstacks -------------------------------------------
    outputs.save_synthetic_centroids(row, p, row_outdir, frames,
                                     site_data, is_synthetic)
    if config.get('save_positions'):
        outputs.save_molecule_positions(
            row_idx, row_outdir, frames, mol_data,
            filename=config.get('positions_filename', 'positions.csv'))
    outputs.build_hyperstacks(row_idx, row_outdir)

    log("%s %s in %.1fs (%d molecules, %d frames -> %s)" % (
        tag, "Phase 1 done" if (config['auto_normalize'] and config['save_conv'])
        else "Complete", time.time() - t0, n_mol, len(frames), row_outdir))
    return row_idx, folder, row_max


# ============================================================================
# PHASE 2: RENDER WITH THE SHARED MAXIMUM (--auto-normalize)
# ============================================================================

def render_normalized_row(args_tuple):
    """Render one row's kept grids with config['auto_vmax'], then drop them."""
    row_idx, row_dict, config = args_tuple
    row = pd.Series(row_dict)
    p = extract_row_params(row)
    row_outdir = os.path.join(config['outdir'], outputs.row_folder(row_idx, p))
    grids_path = os.path.join(row_outdir, '_grids.npz')
    if not os.path.exists(grids_path):
        log("[Row %d] Phase 2: no _grids.npz (already rendered?), skipping" % row_idx)
        return row_idx

    # Same image-noise stream the row would have used in a single pass.
    render_rng = render_rng_from_entropy(outputs.read_seed_entropy(row_outdir))
    vmax = config['auto_vmax']
    bit_depth = config['bit_depth']
    noise_on = p['display_noise_enabled']
    subdir = 'Convolved_auto_%dbit' % bit_depth

    with np.load(grids_path) as npz:
        for frame in sorted(int(k[1:]) for k in npz.files):
            images = render_normalized_slices(
                npz['f%d' % frame], float(row['nucleus_diameter_nm']),
                float(row['voxel_size_nm']), float(row['slice_spacing_nm']),
                vmax, num_slices=p['num_slices'], bit_depth=bit_depth,
                upscale_factor=config['upscale_factor'],
                noise_floor_frac=p['display_noise_floor_pct'] / 100.0 if noise_on else 0.0,
                noise_std_frac=p['display_noise_std_pct'] / 100.0 if noise_on else 0.0,
                rng=render_rng)
            outputs.write_frame_stack(row_outdir, subdir, frame, images)
    outputs.build_hyperstacks(row_idx, row_outdir, only=[subdir])
    os.remove(grids_path)
    log("[Row %d] Phase 2 complete (vmax=%.6f)" % (row_idx, vmax))
    return row_idx


def phase2_vmax(batch_df, outdir, headroom, bit_depth):
    """vmax from EVERY CSV row's _max.txt (SLURM phase 2).

    Raises RuntimeError naming the rows whose phase 1 has not finished.
    _max.txt files are never deleted, so tasks starting at different times
    all compute the same maximum.
    """
    maxima, missing = [], []
    for idx in range(len(batch_df)):
        p = extract_row_params(batch_df.iloc[idx])
        m = outputs.read_row_max(os.path.join(outdir, outputs.row_folder(idx, p)))
        if m is None:
            missing.append(idx)
        else:
            maxima.append(m)
    if missing:
        raise RuntimeError("phase 1 not finished for rows %s (no _max.txt in %s)"
                           % (missing, outdir))
    return outputs.write_calibration(outdir, max(maxima), headroom, bit_depth)
