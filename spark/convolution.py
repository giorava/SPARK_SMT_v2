"""
convolution.py - All-molecule fluorescence images.

1. psf_grid(): every molecule position of every exposure sub-step is
   snapped to its voxel and a normalised 3D Gaussian PSF (cut at 3 sigma)
   is added there. Summing one kernel per point equals convolving the
   point map with the PSF. Light outside the nucleus is removed.
2. The 3D grid is cut into z-planes and scaled in one of two ways:
     render_au_slices()         absolute units: the nuclear mean is set to
                                (conc / ref_conc) * ref_signal, then
                                background and Gaussian noise are added
     render_normalized_slices() divided by a common vmax (auto-normalize),
                                optional display noise
"""

import numpy as np
from PIL import Image

from .geometry import view_grid, slice_z_positions


_FWHM_PER_SIGMA = 2 * np.sqrt(2 * np.log(2))


def psf_grid(positions_um, nucleus_diameter_nm, voxel_size_nm,
             lateral_fwhm_nm, axial_fwhm_nm):
    """Blurred 3D intensity grid from an (M, 3) array of positions in µm."""
    nucleus_radius_nm = nucleus_diameter_nm / 2
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)
    grid_size, half_span = g.grid_size, g.half_span
    coords = positions_um * 1000
    grid_indices = np.floor((coords + half_span) / voxel_size_nm).astype(int)
    grid = np.zeros((grid_size, grid_size, grid_size), dtype=float)

    sigma_xy = max(lateral_fwhm_nm / _FWHM_PER_SIGMA / voxel_size_nm, 0.5)
    sigma_z = max(axial_fwhm_nm / _FWHM_PER_SIGMA / voxel_size_nm, 0.5)
    kr_xy = int(np.ceil(3 * sigma_xy))
    kr_z = int(np.ceil(3 * sigma_z))
    a_xy = np.arange(-kr_xy, kr_xy + 1)
    a_z = np.arange(-kr_z, kr_z + 1)
    xx, yy, zz = np.meshgrid(a_xy, a_xy, a_z, indexing='ij')
    kernel = np.exp(-(xx ** 2 + yy ** 2) / (2 * sigma_xy ** 2) - zz ** 2 / (2 * sigma_z ** 2))
    kernel /= kernel.sum()

    for gx, gy, gz in grid_indices:
        if 0 <= gx < grid_size and 0 <= gy < grid_size and 0 <= gz < grid_size:
            x0, x1 = max(0, gx - kr_xy), min(grid_size, gx + kr_xy + 1)
            y0, y1 = max(0, gy - kr_xy), min(grid_size, gy + kr_xy + 1)
            z0, z1 = max(0, gz - kr_z), min(grid_size, gz + kr_z + 1)
            kx0, kx1 = x0 - (gx - kr_xy), x1 - (gx - kr_xy)
            ky0, ky1 = y0 - (gy - kr_xy), y1 - (gy - kr_xy)
            kz0, kz1 = z0 - (gz - kr_z), z1 - (gz - kr_z)
            if kx1 > kx0 and ky1 > ky0 and kz1 > kz0:
                grid[x0:x1, y0:y1, z0:z1] += kernel[kx0:kx1, ky0:ky1, kz0:kz1]

    xc, yc, zc = np.meshgrid(g.voxel_centers, g.voxel_centers, g.voxel_centers,
                             indexing='ij')
    grid *= (xc ** 2 + yc ** 2 + zc ** 2) <= nucleus_radius_nm ** 2
    return grid


def exposure_psf_grid(exposure_data, frame, nucleus_diameter_nm, voxel_size_nm,
                      lateral_fwhm_nm, axial_fwhm_nm):
    """psf_grid() of all sub-step positions of one frame's exposure window."""
    exp = exposure_data.get(frame)
    if exp is None:
        return None
    positions = exp['pos_steps'].reshape(-1, 3).astype(float)
    return psf_grid(positions, nucleus_diameter_nm, voxel_size_nm,
                    lateral_fwhm_nm, axial_fwhm_nm)


def _bit_depth_info(bit_depth):
    if bit_depth == 16:
        return 65535, np.uint16, 'I;16'
    return 255, np.uint8, 'L'


def _iter_slices(grid, nucleus_diameter_nm, voxel_size_nm, slice_spacing_nm,
                 num_slices):
    """Yield (z_nm, 2D image slice) for each imaged plane, cropped and
    oriented like the saved images (row 0 = top = largest y)."""
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)
    for z_nm in slice_z_positions(nucleus_diameter_nm / 2, slice_spacing_nm, num_slices):
        k = np.clip(int(np.floor((z_nm + g.half_span) / voxel_size_nm)), 0, g.grid_size - 1)
        sl = grid[g.i_start:g.i_end, g.i_start:g.i_end, k].T
        yield z_nm, np.flipud(sl)


def _to_image(arr, bit_depth, upscale_factor):
    max_val, dtype, pil_mode = _bit_depth_info(bit_depth)
    img = np.clip(arr, 0, max_val).astype(dtype)
    if upscale_factor > 1:
        img = np.repeat(np.repeat(img, upscale_factor, axis=0), upscale_factor, axis=1)
    return Image.fromarray(img, mode=pil_mode)


def render_au_slices(grid, nucleus_diameter_nm, voxel_size_nm, slice_spacing_nm,
                     au_calibration, num_slices=None, bit_depth=8,
                     upscale_factor=1, rng=None):
    """Z-planes in absolute units; returns one PIL image per plane.

    au_calibration: dict with ref_concentration, ref_signal,
    sim_concentration, background, noise_std. Pixels outside the nuclear
    cross-section stay 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    nucleus_radius_nm = nucleus_diameter_nm / 2
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)

    xc, yc, zc = np.meshgrid(g.voxel_centers, g.voxel_centers, g.voxel_centers,
                             indexing='ij')
    in_nucleus = grid[(xc ** 2 + yc ** 2 + zc ** 2) <= nucleus_radius_nm ** 2]
    mean_nucleus = np.mean(in_nucleus) if len(in_nucleus) > 0 else 1.0
    if mean_nucleus <= 0:
        mean_nucleus = 1e-10
    ref_conc = au_calibration['ref_concentration']
    expected = ((au_calibration['sim_concentration'] / ref_conc)
                * au_calibration['ref_signal'] if ref_conc > 0 else 0.0)
    scale = expected / mean_nucleus
    background = au_calibration['background']
    noise_std = au_calibration['noise_std']

    cropped = g.voxel_centers[g.i_start:g.i_end]
    mx, my = np.meshgrid(cropped, cropped)
    r_sq = mx ** 2 + my ** 2

    images = []
    for z_nm, sl in _iter_slices(grid, nucleus_diameter_nm, voxel_size_nm,
                                 slice_spacing_nm, num_slices):
        mask = r_sq <= max(0, nucleus_radius_nm ** 2 - z_nm ** 2)
        scaled = np.zeros_like(sl)
        scaled[mask] = sl[mask] * scale + background
        if noise_std > 0:
            noise = rng.normal(0, noise_std, scaled.shape)
            scaled[mask] += noise[mask]
        images.append(_to_image(scaled, bit_depth, upscale_factor))
    return images


def render_normalized_slices(grid, nucleus_diameter_nm, voxel_size_nm,
                             slice_spacing_nm, vmax, num_slices=None,
                             bit_depth=8, upscale_factor=1,
                             noise_floor_frac=0.0, noise_std_frac=0.0, rng=None):
    """Z-planes scaled by grid / vmax (clipped to 0-1); one PIL image per plane.

    Optional display noise with mean noise_floor_frac and std noise_std_frac
    (fractions of full scale) over the whole image.
    """
    if rng is None:
        rng = np.random.default_rng()
    max_val = _bit_depth_info(bit_depth)[0]
    images = []
    for _, sl in _iter_slices(grid, nucleus_diameter_nm, voxel_size_nm,
                              slice_spacing_nm, num_slices):
        normalized = np.clip(sl / vmax, 0.0, 1.0)
        if noise_floor_frac > 0 or noise_std_frac > 0:
            noise = rng.normal(noise_floor_frac, noise_std_frac, normalized.shape)
            normalized = np.clip(normalized + noise, 0.0, 1.0)
        images.append(_to_image(normalized * max_val, bit_depth, upscale_factor))
    return images
