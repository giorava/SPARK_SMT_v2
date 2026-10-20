"""
geometry.py - Image geometry shared by every renderer.

The nucleus is centred at the origin. Images are cut from a cubic voxel
grid of GRID_SIZE voxels per side (odd, so a voxel is centred on 0) and
cropped laterally to the voxels within 1.1 nucleus radii. In the saved
images row 0 is the top (largest y).
"""

from collections import namedtuple

import numpy as np


ViewGrid = namedtuple('ViewGrid', [
    'grid_size',        # voxels per side of the cubic grid
    'half_span',        # nm from the centre to the grid edge
    'voxel_centers',    # (grid_size,) voxel-centre coordinates in nm
    'i_start', 'i_end', # lateral crop [i_start, i_end) in voxel indices
])


def view_grid(nucleus_diameter_nm, voxel_size_nm):
    """Voxel grid and lateral crop for a nucleus of the given diameter."""
    plot_limit = nucleus_diameter_nm / 2 * 1.1
    grid_size = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if grid_size % 2 == 0:
        grid_size += 1
    half_span = (grid_size * voxel_size_nm) / 2.0
    voxel_centers = np.arange(grid_size) * voxel_size_nm - half_span + voxel_size_nm / 2
    in_view = (voxel_centers >= -plot_limit) & (voxel_centers <= plot_limit)
    view_indices = np.where(in_view)[0]
    if len(view_indices) > 0:
        i_start, i_end = view_indices[0], view_indices[-1] + 1
    else:
        i_start, i_end = 0, grid_size
    return ViewGrid(grid_size, half_span, voxel_centers, i_start, i_end)


def slice_z_positions(nucleus_radius_nm, slice_spacing_nm, num_slices=None):
    """Depths (nm) of the imaged z-planes, sorted, inside the nucleus.

    With num_slices: planes at multiples of slice_spacing_nm, centred on 0,
    keeping only those inside the nucleus. Without: every multiple of the
    spacing inside the nucleus.
    """
    if num_slices is not None:
        n_half = int(num_slices) // 2
        cand = sorted({i * slice_spacing_nm for i in range(-n_half, n_half + 1)})
        return [z for z in cand if abs(z) <= nucleus_radius_nm]
    return sorted(np.unique(np.concatenate([
        np.arange(0, nucleus_radius_nm, slice_spacing_nm),
        -np.arange(0, nucleus_radius_nm, slice_spacing_nm)])))


def synthetic_centroid_pixels(sites_pos, is_synthetic, frame_idx,
                              nucleus_diameter_nm, voxel_size_nm):
    """Centroid of the synthetic sites in image pixels for one frame.

    Returns {'Frame', 'Z', 'Y', 'X'} or None if there are no synthetic
    sites. X/Y are pixels of the saved image; Z is the voxel index of the
    full grid (not a slice number).
    """
    if is_synthetic is None or not np.any(is_synthetic):
        return None
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)
    centroid_nm = np.mean(sites_pos[is_synthetic] * 1000, axis=0)
    grid_x, grid_y, grid_z = (centroid_nm + g.half_span) / voxel_size_nm
    return {'Frame': frame_idx, 'Z': grid_z,
            'Y': (g.i_end - 1) - grid_y, 'X': grid_x - g.i_start}
