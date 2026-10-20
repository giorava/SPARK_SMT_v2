"""
positions.py - Positions of labelled molecules and of the synthetic array.

Positions in the tracker result and in exposure_data are in µm; everything
returned here is in nm.
"""

import numpy as np


def exposure_window_positions(exposure_data, frame, mol_ids):
    """(S, L, 3) positions in µm during the exposure of `frame`, or None."""
    exp = exposure_data.get(frame) if exposure_data is not None else None
    if exp is None:
        return None
    return np.asarray(exp['pos_steps'])[:, mol_ids, :]


def labeled_positions(res, exposure_data, mol_ids, n_steps, spd, spe):
    """(n_steps, L, 3) positions in nm; NaN where not recorded.

    Sources, later ones overwriting earlier ones:
      1. exposure windows (always available: what the camera saw)
      2. full trajectory dump (every `trajectory_stride` steps)
      3. molecules watched by the tracker (every step)
    """
    mol_ids = np.asarray(mol_ids, dtype=np.int64)
    out = np.full((n_steps, len(mol_ids), 3), np.nan)
    for f in range(n_steps // spd + 1):
        P = exposure_window_positions(exposure_data, f, mol_ids)
        if P is None:
            continue
        k0 = f * spd
        m = min(len(P), n_steps - k0)
        if m > 0:
            out[k0:k0 + m] = P[:m] * 1000.0
    if 'traj_step' in res:
        ok = res['traj_step'] < n_steps
        out[res['traj_step'][ok]] = res['traj_pos'][ok][:, mol_ids, :] * 1000.0
    w = list(res['watch_ids'])
    if w:
        for j, m in enumerate(mol_ids):
            if m in w:
                out[:, j, :] = res['watch_pos'][:n_steps, w.index(m), :] * 1000.0
    return out


def _exposure_slices(res, num_frames, spd, spe):
    """Step range [k0, k1) of each frame's exposure in res['synth_pos']."""
    last = len(res['synth_pos']) - 1
    for f in range(int(num_frames)):
        k0 = min(f * spd, last)
        yield f, k0, min(k0 + max(int(spe), 1), last + 1)


def array_centroid_per_frame(res, num_frames, spd, spe):
    """(F, 3) synthetic-array centroid in nm, averaged over all synthetic
    sites and over the exposure window of each frame. None if no array."""
    if len(res['synthetic_site_ids']) == 0:
        return None
    out = np.empty((int(num_frames), 3))
    for f, k0, k1 in _exposure_slices(res, num_frames, spd, spe):
        out[f] = res['synth_pos'][k0:k1].reshape(-1, 3).mean(axis=0) * 1000.0
    return out


def array_depth_per_frame(res, num_frames, spd, spe):
    """(F,) mean depth (nm) of the synthetic array during each frame's
    exposure: the focal plane of --smt-focus track. None if no array."""
    if len(res['synthetic_site_ids']) == 0:
        return None
    out = np.empty(int(num_frames))
    for f, k0, k1 in _exposure_slices(res, num_frames, spd, spe):
        out[f] = res['synth_pos'][k0:k1, :, 2].mean() * 1000.0
    return out
