"""
outputs.py - Everything written to a row's Tracking/ folder (plus the SMT
images), driven by write_tracking_outputs().

Tracking/
  state_events.csv            every state / binding change of every molecule
  site_dwells.csv             one row per binding interval
  synthetic_dwells.csv        the intervals on the synthetic array
  synthetic_visits.csv        back-to-back intervals merged into visits
  labeled_track.csv           labelled molecules at every physics step
  labeled_frames.csv          labelled molecules per camera frame
  synthetic_neighborhood.csv  molecules near the array, per frame
  smt_focus_track.csv         focal depth per frame (--smt-focus track)
  tracking_arrays.npz         raw arrays (array positions, occupants, ...)
  tracking_summary.json       labelled ids, counts, settings
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from spark.outputs import STATE_LABELS
from .labels import resolve_labels
from .events import events_dataframe, dwell_table, merge_visits, state_timeline
from .geometry import to_pixels, with_focal_plane
from .positions import (labeled_positions, array_centroid_per_frame,
                        array_depth_per_frame)
from .images import write_smt_images


# ============================================================================
# FOCAL PLANE
# ============================================================================

def choose_focus(res, focus_mode, geom, num_frames, spd, spe, log=print):
    """Apply --smt-focus. Returns (geom, focus_z).

    'slices'    geom unchanged, focus_z None
    'synthetic' one fixed plane at the array depth at t = 0
    'track'     focus_z = per-frame array depth (focus lock); geom gets the
                first frame's plane
    Without synthetic sites the modes fall back to 'slices'.
    """
    has_array = len(res['synthetic_site_ids']) > 0
    if focus_mode == 'track':
        if not has_array:
            log("  Tracking: --smt-focus track ignored (no synthetic sites)")
            return geom, None
        focus_z = array_depth_per_frame(res, num_frames, spd, spe)
        log("  Tracking: SMT focus follows the synthetic array, z = %.0f "
            "to %.0f nm over the movie" % (focus_z.min(), focus_z.max()))
        return with_focal_plane(geom, focus_z[0]), focus_z
    if focus_mode == 'synthetic':
        if not has_array:
            log("  Tracking: --smt-focus synthetic ignored (no synthetic sites)")
            return geom, None
        z_focus = float(res['synth_pos'][0, :, 2].mean() * 1000.0)
        log("  Tracking: SMT focal plane set to synthetic array depth "
            "z = %.0f nm (fixed; the array may drift afterwards)" % z_focus)
        return with_focal_plane(geom, z_focus), None
    return geom, None


# ============================================================================
# TABLES
# ============================================================================

def synthetic_occupancy(res, syn_dwells):
    """(final_step + 1, n_synthetic) id of the molecule on each synthetic
    site at each step, -1 when empty."""
    occ = np.full((res['final_step'] + 1, len(res['synthetic_site_ids'])), -1,
                  dtype=np.int32)
    for r in syn_dwells.itertuples(index=False):
        occ[r.bind_step:r.unbind_step, r.synthetic_index] = r.molecule_id
    return occ


def labeled_track_table(res, exposure_data, labeled, n_steps, spd, spe, geom,
                        focus_z=None):
    """Labelled molecules at every physics step: state, bound site,
    position (nm and px), distance to the focal plane and to the array."""
    dt = res['sim_dt_s']
    st, bs = state_timeline(res, labeled, n_steps)
    P = labeled_positions(res, exposure_data, labeled, n_steps, spd, spe)
    x_px, y_px, zk, dz = to_pixels(P, geom)
    steps = np.arange(n_steps)
    if focus_z is not None:
        fz_step = focus_z[np.minimum(steps // spd, len(focus_z) - 1)]
        dz = P[:, :, 2] - fz_step[:, None]
        zk = np.zeros(dz.shape, dtype=int)
    synth_nm = res['synth_pos'][:n_steps].astype(float) * 1000.0
    if len(res['synthetic_site_ids']):
        d = np.linalg.norm(P[:, :, None, :] - synth_nm[:, None, :, :], axis=-1)
        dist_syn = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=-1)
        dist_syn[np.isnan(P[:, :, 0])] = np.nan
    else:
        dist_syn = np.full(st.shape, np.nan)
    is_syn = res['is_synthetic']
    on_syn = np.where(bs >= 0, is_syn[np.maximum(bs, 0)], False)
    L = len(labeled)
    track = pd.DataFrame({
        'step': np.repeat(steps, L),
        'time_s': np.repeat(steps * dt, L),
        'frame': np.repeat(steps // spd, L),
        'in_exposure': np.repeat((steps % spd) < spe, L),
        'molecule_id': np.tile(labeled, n_steps),
        'state': st.ravel(),
        'state_label': pd.Series(st.ravel()).map(STATE_LABELS).values,
        'bound_site': bs.ravel(),
        'bound_to_synthetic': on_syn.ravel(),
        'x_nm': P[:, :, 0].ravel(), 'y_nm': P[:, :, 1].ravel(),
        'z_nm': P[:, :, 2].ravel(),
        'x_px': x_px.ravel(), 'y_px': y_px.ravel(),
        'nearest_slice': np.where(np.isnan(P[:, :, 2]), -1, zk).ravel(),
        'dz_to_slice_nm': dz.ravel(),
        'dist_to_synthetic_nm': dist_syn.ravel(),
    })
    if focus_z is not None:
        track['focal_z_nm'] = np.repeat(fz_step, L)
    return track


def labeled_frames_table(track, res, geom, focus_z, num_frames, spd, spe):
    """Labelled molecules per camera frame (averages over the exposure),
    plus the array centroid in pixels."""
    dt = res['sim_dt_s']
    ex = track[track['in_exposure']]
    g = ex.groupby(['frame', 'molecule_id'], sort=True)
    frames_df = g.agg(x_nm=('x_nm', 'mean'), y_nm=('y_nm', 'mean'),
                      z_nm=('z_nm', 'mean'),
                      frac_bound=('state', lambda s: float(np.mean(s == 2))),
                      frac_on_synthetic=('bound_to_synthetic', 'mean'),
                      min_dist_to_synthetic_nm=('dist_to_synthetic_nm', 'min'),
                      modal_state=('state', lambda s: int(pd.Series(s).mode().iloc[0])),
                      ).reset_index()
    frames_df.insert(1, 'time_s', frames_df['frame'] * spd * dt)
    fx, fy, fk, fdz = to_pixels(frames_df[['x_nm', 'y_nm', 'z_nm']].values, geom)
    if focus_z is not None:
        _fz = focus_z[np.minimum(frames_df['frame'].values, len(focus_z) - 1)]
        fdz = frames_df['z_nm'].values - _fz
        fk = np.zeros(len(frames_df), dtype=int)
        frames_df['focal_z_nm'] = _fz
    frames_df['x_px'], frames_df['y_px'] = fx, fy
    frames_df['nearest_slice'], frames_df['dz_to_slice_nm'] = fk, fdz
    frames_df['modal_state_label'] = frames_df['modal_state'].map(STATE_LABELS)
    arr_c = array_centroid_per_frame(res, num_frames, spd, spe)
    if arr_c is not None:
        _ac = arr_c[np.minimum(frames_df['frame'].values, len(arr_c) - 1)]
        ax_px, ay_px, _, _ = to_pixels(_ac, geom)
        frames_df['array_x_px'], frames_df['array_y_px'] = ax_px, ay_px
        frames_df['array_z_nm'] = _ac[:, 2]
    return frames_df


def neighborhood_table(res, exposure_data, labeled, syn_occ, num_frames, spd,
                       near_radius_nm):
    """Per frame: molecules within near_radius_nm of any synthetic site
    (mean / max over the exposure sub-steps) and who occupies the array."""
    dt = res['sim_dt_s']
    rows = []
    for f in range(int(num_frames)):
        exp = exposure_data.get(f)
        if exp is None:
            continue
        Pall = np.asarray(exp['pos_steps']) * 1000.0
        k0 = f * spd
        n_near, lab_near = [], []
        for j in range(len(Pall)):
            k = min(k0 + j, len(res['synth_pos']) - 1)
            tree = cKDTree(res['synth_pos'][k].astype(float) * 1000.0)
            dd, _ = tree.query(Pall[j], k=1)
            near = dd <= near_radius_nm
            n_near.append(int(near.sum()))
            lab_near.append(int(near[labeled].sum()))
        occ = syn_occ[k0:k0 + len(Pall)]
        occupants = sorted(set(occ[occ >= 0].tolist()))
        rows.append({
            'frame': f, 'time_s': k0 * dt,
            'mean_molecules_within_r': float(np.mean(n_near)),
            'max_molecules_within_r': int(np.max(n_near)),
            'labeled_within_r_any': bool(np.any(lab_near)),
            'mean_synthetic_sites_occupied': float(np.mean((occ >= 0).sum(axis=1))),
            'occupant_ids': ';'.join(str(o) for o in occupants),
            'labeled_occupant': bool(np.isin(occupants, labeled).any()) if occupants else False,
        })
    return pd.DataFrame(rows)


def write_arrays(tdir, res, syn_occ, labeled, focus_z, spd):
    """tracking_arrays.npz (and smt_focus_track.csv in track mode)."""
    dt = res['sim_dt_s']
    npz = {'synthetic_site_ids': res['synthetic_site_ids'],
           'synthetic_pos_nm': res['synth_pos'].astype(np.float32) * 1000.0,
           'synthetic_occupant': syn_occ,
           'labeled_ids': labeled,
           'is_synthetic': res['is_synthetic'],
           'site_strengths': res['site_strengths']}
    if focus_z is not None:
        npz['smt_focus_z_nm'] = focus_z
        pd.DataFrame({'frame': np.arange(len(focus_z)),
                      'time_s': np.arange(len(focus_z)) * spd * dt,
                      'focal_z_nm': focus_z}).to_csv(
            os.path.join(tdir, 'smt_focus_track.csv'), index=False,
            float_format='%.3f')
    if 'traj_step' in res:
        npz.update(traj_step=res['traj_step'],
                   traj_pos_nm=res['traj_pos'] * np.float32(1000.0),
                   traj_states=res['traj_states'], traj_bound=res['traj_bound'])
    np.savez(os.path.join(tdir, 'tracking_arrays.npz'), **npz)


# ============================================================================
# MAIN WRITER
# ============================================================================

def write_tracking_outputs(row_outdir, res, exposure_data, label_spec,
                           label_rng, geom, num_frames, spd, spe,
                           lateral_fwhm_nm, axial_fwhm_nm,
                           near_radius_nm=250.0, write_smt=False,
                           smt_params=None, bit_depth=16, log=print):
    """Write binding tables, labelled tracks and (optionally) SMT images.

    res        BindingTracker.result()
    label_rng  chooses labels, then draws the SMT camera noise
    geom       image_geometry() of the row
    spd / spe  physics steps per display frame / per exposure
    Returns the array of labelled molecule ids.
    """
    tdir = os.path.join(row_outdir, 'Tracking')
    os.makedirs(tdir, exist_ok=True)
    dt = res['sim_dt_s']
    n_steps = min(int(num_frames) * int(spd), res['final_step'])

    focus_mode = (smt_params or {}).get('focus', 'slices') if write_smt else 'slices'
    geom, focus_z = choose_focus(res, focus_mode, geom, num_frames, spd, spe, log)

    # ---- events & dwells (all molecules) --------------------------------
    dwells = dwell_table(res)
    labeled, note = resolve_labels(res, label_spec, label_rng, dwells)
    if note:
        log("  Tracking: " + note)
    dwells['labeled'] = dwells['molecule_id'].isin(labeled)
    events_dataframe(res, labeled).to_csv(
        os.path.join(tdir, 'state_events.csv'), index=False)
    dwells.to_csv(os.path.join(tdir, 'site_dwells.csv'), index=False)
    syn_dw = dwells[dwells['is_synthetic']]
    syn_dw.to_csv(os.path.join(tdir, 'synthetic_dwells.csv'), index=False)
    visits = merge_visits(dwells, max_gap_steps=0, sim_dt_s=dt)
    visits.to_csv(os.path.join(tdir, 'synthetic_visits.csv'), index=False)
    syn_occ = synthetic_occupancy(res, syn_dw)

    # ---- labelled molecules ---------------------------------------------
    track = labeled_track_table(res, exposure_data, labeled, n_steps, spd, spe,
                                geom, focus_z)
    track.to_csv(os.path.join(tdir, 'labeled_track.csv'), index=False,
                 float_format='%.3f')
    labeled_frames_table(track, res, geom, focus_z, num_frames, spd, spe).to_csv(
        os.path.join(tdir, 'labeled_frames.csv'), index=False, float_format='%.3f')

    # ---- neighbourhood of the synthetic array, raw arrays ---------------
    if len(res['synthetic_site_ids']) and exposure_data is not None:
        neighborhood_table(res, exposure_data, labeled, syn_occ, num_frames, spd,
                           near_radius_nm).to_csv(
            os.path.join(tdir, 'synthetic_neighborhood.csv'), index=False,
            float_format='%.3f')
    write_arrays(tdir, res, syn_occ, labeled, focus_z, spd)

    # ---- SMT images -----------------------------------------------------
    display_range, cross_on = None, False
    if write_smt:
        display_range, cross_on = write_smt_images(
            row_outdir, res, exposure_data, labeled, geom, focus_z,
            num_frames, spd, spe, lateral_fwhm_nm, axial_fwhm_nm,
            smt_params, bit_depth, label_rng, log)

    summary = {
        'label_spec': str(label_spec),
        'labeled_ids': [int(i) for i in labeled],
        'label_note': note,
        'num_molecules': int(res['num_molecules']),
        'num_sites': int(res['num_sites']),
        'synthetic_site_ids': [int(i) for i in res['synthetic_site_ids']],
        'sim_dt_s': dt, 'steps_per_display': int(spd),
        'steps_per_exposure': int(spe), 'num_frames': int(num_frames),
        'n_binding_intervals': int(len(dwells)),
        'n_synthetic_binding_intervals': int(len(syn_dw)),
        'n_labeled_synthetic_binding_intervals': int(syn_dw['labeled'].sum()),
        'n_synthetic_visits_merged': int(len(visits)),
        'n_labeled_synthetic_visits_merged': int(visits['labeled'].sum()),
        'synthetic_binder_ids': sorted(int(i) for i in syn_dw['molecule_id'].unique()),
        'image_geometry': geom,
        'smt_rendered': bool(write_smt),
        'smt_params': smt_params if write_smt else None,
        'smt_focus_mode': focus_mode if (focus_mode != 'track' or focus_z is not None) else 'slices',
        'smt_focus_z_range_nm': ([float(focus_z.min()), float(focus_z.max())]
                                 if focus_z is not None else None),
        'smt_display_range_adu': display_range,
        'smt_bit_depth': (8 if int(bit_depth) == 8 else 16) if write_smt else None,
        'smt_cross_drawn': cross_on,
        'near_radius_nm': near_radius_nm,
        'pixel_convention': ('pixel (c, r) covers [c, c+1) x [r, r+1); '
                             'centre at +0.5; row 0 = top (largest y)'),
    }
    with open(os.path.join(tdir, 'tracking_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    log("  Tracking: labelled %s | %d binding intervals (%d on synthetic "
        "array, %d by labelled) -> %s" % (
            summary['labeled_ids'], summary['n_binding_intervals'],
            summary['n_synthetic_binding_intervals'],
            summary['n_labeled_synthetic_binding_intervals'], tdir))
    return labeled
