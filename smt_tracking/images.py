"""
images.py - Writing the SMT image stacks (SMT_<bit>bit/).

One TIFF z-stack per frame. A single display range is used for the whole
movie; 16-bit stacks keep raw camera ADU (range embedded for ImageJ),
8-bit stacks are contrast-stretched with that range. Optionally a '+' marks
the synthetic-array centroid on the planes near the array.
"""

import os

import numpy as np
from tifffile import imwrite

from .geometry import to_pixels, with_focal_plane
from .positions import exposure_window_positions, array_centroid_per_frame
from .rendering import render_smt_frame, draw_open_cross


def write_smt_images(row_outdir, res, exposure_data, labeled, geom, focus_z,
                     num_frames, spd, spe, lateral_fwhm_nm, axial_fwhm_nm,
                     smt_params, bit_depth, rng, log=print):
    """Render and save the SMT frames.

    focus_z: per-frame focal depth (nm) for --smt-focus track, else None
    (then geom's z-planes are used). rng draws the camera noise.
    Returns (display_range or None, cross_drawn).
    """
    sp = dict(smt_params or {})
    bit_depth = 8 if int(bit_depth) == 8 else 16
    sdir = os.path.join(row_outdir, 'SMT_%dbit' % bit_depth)
    os.makedirs(sdir, exist_ok=True)

    stacks, frame_geoms = {}, {}
    for f in range(int(num_frames)):
        W = exposure_window_positions(exposure_data, f, labeled)
        if W is None:
            continue
        frame_geoms[f] = (with_focal_plane(geom, focus_z[f])
                          if focus_z is not None else geom)
        stacks[f] = render_smt_frame(
            W * 1000.0, frame_geoms[f], lateral_fwhm_nm, axial_fwhm_nm,
            photons=sp.get('photons', 300.0),
            background=sp.get('background', 10.0),
            gain=sp.get('gain', 1.0), offset=sp.get('offset', 100.0),
            read_noise=sp.get('read_noise', 1.5),
            add_noise=sp.get('add_noise', True), rng=rng)
    if not stacks:
        return None, False

    # One display range for the whole movie, computed from the rendered data
    # BEFORE any marker is drawn.
    allpx = np.concatenate([s.ravel() for s in stacks.values()])
    lo = float(np.percentile(allpx, 0.5))
    hi = float(allpx.max())
    if hi <= lo:
        hi = lo + 1.0
    arr_c = array_centroid_per_frame(res, num_frames, spd, spe)
    cross_on = bool(sp.get('cross', True)) and arr_c is not None

    for f, stack in stacks.items():
        if bit_depth == 8:
            # Contrast-stretched with the global range (not raw ADU).
            out = np.rint(np.clip((stack.astype(float) - lo) / (hi - lo),
                                  0, 1) * 255).astype(np.uint8)
            mark, meta = 255, {'axes': 'ZYX'}
        else:
            out = stack.copy()
            mark, meta = int(hi), {'axes': 'ZYX', 'min': lo, 'max': hi}
        if cross_on:
            _draw_array_cross(out, arr_c[min(f, len(arr_c) - 1)], geom,
                              frame_geoms[f]['z_slices_nm'], axial_fwhm_nm, mark)
        imwrite(os.path.join(sdir, 'frame_%04d.tiff' % f), out,
                imagej=True, metadata=meta, compression='zlib')

    log("  Tracking: SMT images -> SMT_%dbit/ (%s%s)" % (
        bit_depth,
        'raw camera ADU %d-%d, display range embedded' % (int(allpx.min()), int(hi))
        if bit_depth == 16 else
        'contrast-stretched from %d-%d ADU; use --bit-depth 16 for raw counts' % (int(lo), int(hi)),
        '; array centroid marked with a cross' if cross_on else ''))
    return (lo, hi), cross_on


def _draw_array_cross(stack, centroid_nm, geom, z_slices_nm, axial_fwhm_nm, value):
    """Mark the array centroid on every plane within one axial FWHM of it
    (or on the nearest plane if none is that close)."""
    cx, cy, _, _ = to_pixels(centroid_nm, geom)
    zs = np.asarray(z_slices_nm, dtype=float)
    if not len(zs):
        return
    near = np.abs(zs - centroid_nm[2]) <= axial_fwhm_nm
    if not near.any():
        near[np.argmin(np.abs(zs - centroid_nm[2]))] = True
    for zi in np.where(near)[0]:
        draw_open_cross(stack[zi], cx, cy, value)
