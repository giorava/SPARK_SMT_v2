"""
rendering.py - SMT camera images of the labelled molecules.

The PSF is integrated over each pixel at the exact sub-pixel position of
every labelled molecule at every physics sub-step of the exposure (no voxel
snapping), so motion blur and the localisation ground truth are exact.
Unlabelled molecules emit no light. The camera adds Poisson shot noise,
read noise, gain and offset.
"""

import numpy as np
from scipy.special import erf


_SQRT2 = np.sqrt(2.0)
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


def render_smt_frame(pos_nm, geom, lateral_fwhm_nm, axial_fwhm_nm,
                     photons, background, gain=1.0, offset=100.0,
                     read_noise=1.5, add_noise=True, rng=None):
    """Render one frame (z-stack) of labelled molecules as a camera image.

    pos_nm : (S, L, 3) positions of L labelled molecules at the S physics
             sub-steps of the exposure (nm).
    photons: photons per molecule per exposure when in focus (split evenly
             across sub-steps; axial defocus attenuates as a Gaussian, the
             same separable PSF model the rest of SPARK uses).
    background: photons per pixel per exposure.
    Returns uint16 array (Z, Y, X) in camera ADU.
    """
    n = geom['n_px']
    v = geom['voxel_size_nm']
    zs = np.asarray(geom['z_slices_nm'], dtype=float)
    sxy = lateral_fwhm_nm * _FWHM_TO_SIGMA
    sz = axial_fwhm_nm * _FWHM_TO_SIGMA
    xe = geom['x_edge0_nm'] + np.arange(n + 1) * v           # left -> right
    ye = geom['y_edge_top_nm'] - np.arange(n + 1) * v        # top  -> bottom

    img = np.zeros((len(zs), n, n), dtype=float)
    P = np.asarray(pos_nm, dtype=float).reshape(-1, 3)
    if len(P):
        S = max(1, np.asarray(pos_nm).shape[0])
        cx = 0.5 * (1 + erf((xe[None, :] - P[:, :1]) / (_SQRT2 * sxy)))
        cy = 0.5 * (1 + erf((ye[None, :] - P[:, 1:2]) / (_SQRT2 * sxy)))
        ax = np.diff(cx, axis=1)          # (M, n) fraction per column
        ay = -np.diff(cy, axis=1)         # (M, n) fraction per row
        w = (photons / S) * np.exp(-(P[:, 2][None, :] - zs[:, None]) ** 2
                                   / (2 * sz ** 2))            # (Z, M)
        for zi in range(len(zs)):
            img[zi] = (ay.T * w[zi]) @ ax
    expected = img + background
    if add_noise:
        rng = rng if rng is not None else np.random.default_rng()
        electrons = rng.poisson(expected) + rng.normal(0.0, read_noise, expected.shape)
        adu = gain * electrons + offset
    else:
        adu = gain * expected + offset
    return np.clip(np.rint(adu), 0, 65535).astype(np.uint16)


def draw_open_cross(img2d, x_px, y_px, value, gap=2, arm=3):
    """Draw a '+' marker centred on continuous pixel coords (x_px, y_px).

    The centre is left open (pixels closer than `gap` are untouched) so a
    molecule sitting on the array is not hidden. Arms are `arm` px long.
    Modifies img2d in place.
    """
    if not (np.isfinite(x_px) and np.isfinite(y_px)):
        return
    n_r, n_c = img2d.shape
    c, r = int(np.floor(x_px)), int(np.floor(y_px))
    for d in range(gap, gap + arm):
        for rr, cc in ((r, c - d), (r, c + d), (r - d, c), (r + d, c)):
            if 0 <= rr < n_r and 0 <= cc < n_c:
                img2d[rr, cc] = value
