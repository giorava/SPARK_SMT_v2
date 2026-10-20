"""
outputs.py - Files written per row and per run.

TIFF stacks and ImageJ hyperstacks, ground-truth positions, the synthetic
array centroid, run_info.json (random stream of the row) and the
auto-normalize bookkeeping (_max.txt, brightness_calibration.json).
"""

import glob
import io
import json
import os

import numpy as np
import pandas as pd
from tifffile import imread as tiff_imread, imwrite as tiff_imwrite

from .geometry import synthetic_centroid_pixels
from .log import log


def row_folder(row_idx, p):
    """Output folder name of a row: Row_<i>_<label> (or Row_<i>)."""
    return "Row_%d_%s" % (row_idx, p['row_label']) if p['row_label'] else "Row_%d" % row_idx


def write_tiff_stack(images, path):
    """Save a list of PIL images as one multi-page TIFF."""
    tiff_buf = io.BytesIO()
    images[0].save(tiff_buf, format='TIFF', save_all=True,
                   append_images=images[1:], compression='tiff_deflate')
    with open(path, 'wb') as f:
        f.write(tiff_buf.getvalue())


def write_frame_stack(row_outdir, subdir, frame, images):
    """<row_outdir>/<subdir>/frame_<frame>.tiff"""
    d = os.path.join(row_outdir, subdir)
    os.makedirs(d, exist_ok=True)
    write_tiff_stack(images, os.path.join(d, 'frame_%04d.tiff' % frame))


# ---- run_info.json -----------------------------------------------------

def write_run_info(row_outdir, row_idx, base_seed, seed_sequence, shared_sites):
    """Record the row's random stream so it can be re-run identically."""
    with open(os.path.join(row_outdir, 'run_info.json'), 'w') as f:
        json.dump({'row_index': int(row_idx), 'base_seed': base_seed,
                   'seed_entropy': seed_sequence.entropy,
                   'shared_sites': bool(shared_sites)}, f, indent=2)


def read_seed_entropy(row_outdir):
    with open(os.path.join(row_outdir, 'run_info.json')) as f:
        return json.load(f)['seed_entropy']


# ---- auto-normalize bookkeeping ----------------------------------------

def write_row_max(row_outdir, row_max):
    with open(os.path.join(row_outdir, '_max.txt'), 'w') as f:
        f.write('%.10f\n' % row_max)


def read_row_max(row_outdir):
    """The row's maximum, or None if phase 1 has not finished."""
    path = os.path.join(row_outdir, '_max.txt')
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return float(f.read().strip())


def write_calibration(outdir, global_max, headroom, bit_depth):
    """brightness_calibration.json, written atomically (SLURM tasks race).
    Returns vmax = global_max * headroom."""
    auto_vmax = global_max * headroom
    path = os.path.join(outdir, 'brightness_calibration.json')
    tmp = path + '.tmp.%d' % os.getpid()
    with open(tmp, 'w') as f:
        json.dump({'global_max': global_max, 'headroom': headroom,
                   'auto_vmax': auto_vmax, 'bit_depth': bit_depth}, f, indent=2)
    os.replace(tmp, path)
    return auto_vmax


# ---- ground-truth tables -----------------------------------------------

#  State-code -> readable label mapping. Matches the engine's state encoding:
#    0 = free diffusing
#    1 = intermediate  (only reachable when two_state_mode is False)
#    2 = bound to a specific site (reachable in either mode)
#    3 = non-specifically bound
#  `compute_exposure_frames` writes the MODAL state across exposure sub-steps
#  into the 'state' column of mol_data[frame].
STATE_LABELS = {0: 'free', 1: 'intermediate', 2: 'bound', 3: 'nonspecific'}


def save_molecule_positions(row_idx, row_outdir, display_frames,
                             mol_data, filename='positions.csv'):
    """
    Write per-molecule positions across all frames to a ViSP-compatible file,
    with an extra column tagging each molecule's binding state.

    Format (tab-delimited, no header, one localization per line):
        x[nm]\ty[nm]\tz[nm]\tintensity\tframe\tstate

    - Positions are averaged over the exposure window; state is the modal
      state over that window.
    - Intensity is a placeholder of 1; frames are 1-indexed (ViSP).
    - Molecules are written in the same order in every frame, so the n-th
      line of each frame block is molecule n.
    """
    out_path = os.path.join(row_outdir, filename)
    n_rows_written = 0
    with open(out_path, 'w') as f:
        for frame in display_frames:
            mol_df = mol_data.get(frame)
            if mol_df is None or len(mol_df) == 0:
                continue
            # µm -> nm
            xs = mol_df['x'].values * 1000.0
            ys = mol_df['y'].values * 1000.0
            zs = mol_df['z'].values * 1000.0
            states = mol_df['state'].values.astype(int)
            visp_frame = int(frame) + 1   # 1-indexed for ViSP
            for x_nm, y_nm, z_nm, st in zip(xs, ys, zs, states):
                label = STATE_LABELS.get(int(st), 'unknown')
                f.write("%.3f\t%.3f\t%.3f\t%d\t%d\t%s\n" % (
                    x_nm, y_nm, z_nm, 1, visp_frame, label))
                n_rows_written += 1
    log("[Row %d] Positions written: %s (%d localizations, %d frames)" % (
        row_idx, filename, n_rows_written, len(display_frames)))


def save_synthetic_centroids(row, p, row_outdir, display_frames,
                              site_data, is_synthetic):
    if is_synthetic is None or not np.any(is_synthetic) or p['num_synthetic'] <= 0:
        return
    centroid_data = []
    for frame in display_frames:
        if frame in site_data:
            sites_pos, _ = site_data[frame]
            centroid_data.append(synthetic_centroid_pixels(
                sites_pos, is_synthetic, frame,
                float(row['nucleus_diameter_nm']), float(row['voxel_size_nm'])))
    if centroid_data:
        df_c = pd.DataFrame(centroid_data)
        for col in ['Z', 'Y', 'X']:
            df_c[col] = df_c[col].round(3)
        df_c.to_csv(os.path.join(row_outdir, 'synthetic_site_centroid_TZYX.csv'), index=False)


# ---- hyperstacks ---------------------------------------------------------

def build_hyperstacks(row_idx, row_outdir, only=None):
    """
    Combine per-frame z-stack TIFFs into a single hyperstack per subfolder
    (or only the subfolders named in `only`).

    Frame files can be either:
      - grayscale        (Z, Y, X)           → hyperstack (T, Z, Y, X), axes TZYX
      - RGB/RGBA ground  (Z, Y, X, C)        → hyperstack (T, Z, C, Y, X), axes TZCYX
    ImageJ hyperstack convention requires the channel axis between Z and Y,
    which is why we transpose for the 5-D case.

    Each subdir is handled independently — if one fails (e.g. an unexpected
    shape), others still get built and the row finishes.
    """
    for subdir in sorted(os.listdir(row_outdir)):
        if only is not None and subdir not in only:
            continue
        sub_path = os.path.join(row_outdir, subdir)
        if not os.path.isdir(sub_path):
            continue
        frame_files = sorted(glob.glob(os.path.join(sub_path, 'frame_*.tiff')))
        if not frame_files:
            continue
        try:
            stacks = []
            for ff in frame_files:
                zstack = tiff_imread(ff)
                if zstack.ndim == 2:
                    zstack = zstack[np.newaxis, ...]   # promote (Y,X) → (1,Y,X)
                stacks.append(zstack)
            hyperstack = np.stack(stacks, axis=0)

            if hyperstack.ndim == 4:
                # (T, Z, Y, X) — grayscale convolved images.
                axes = 'TZYX'
                out_arr = hyperstack
            elif hyperstack.ndim == 5:
                # (T, Z, Y, X, C) — RGB(A) ground truth. ImageJ wants
                # channels between Z and Y, so transpose to (T, Z, C, Y, X).
                out_arr = np.transpose(hyperstack, (0, 1, 4, 2, 3))
                axes = 'TZCYX'
            else:
                log("[Row %d] Skipping hyperstack for %s: unexpected ndim=%d "
                     "(shape=%s)" % (row_idx, subdir, hyperstack.ndim,
                                     hyperstack.shape))
                continue

            out_name = '%s_Hyperstack.tif' % subdir
            out_path = os.path.join(row_outdir, out_name)
            _meta = {'axes': axes}
            if out_arr.dtype == np.uint16 and axes == 'TZYX':
                # 16-bit data rarely spans 0-65535; without a stored display
                # range most viewers show it as black. Data is unchanged.
                _lo = float(np.percentile(out_arr, 0.5))
                _hi = float(out_arr.max())
                _meta.update(min=_lo, max=_hi if _hi > _lo else _lo + 1.0)
            tiff_imwrite(out_path, out_arr, imagej=True, metadata=_meta)
            log("[Row %d] Hyperstack: %s (%s, axes=%s)" % (
                row_idx, out_name,
                'x'.join(str(s) for s in out_arr.shape), axes))
        except Exception as e:
            # Don't let one malformed subdir kill the others.
            log("[Row %d] Hyperstack FAILED for %s: %s" % (row_idx, subdir, e))
            continue
