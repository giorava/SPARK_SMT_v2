"""
geometry.py - Pixel coordinates of the SMT images.

The SMT images use exactly the grid, crop and z-planes of SPARK's
all-molecule images (spark.geometry), so the two can be overlaid.

PIXEL CONVENTION
x_px / y_px are continuous coordinates in the saved (non-upscaled) image:
pixel (column c, row r) covers [c, c+1) x [r, r+1), so its centre is at
(c + 0.5, r + 0.5). Row 0 is the top of the image (largest y).
"""

import numpy as np

from spark.geometry import view_grid, slice_z_positions


def image_geometry(nucleus_diameter_nm, voxel_size_nm, slice_spacing_nm,
                   num_slices=None):
    """Image geometry as a JSON-friendly dict: pixel size, image size,
    position of the image edges (nm) and depths of the z-planes (nm)."""
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)
    z = slice_z_positions(nucleus_diameter_nm / 2.0, slice_spacing_nm, num_slices)
    return {
        'voxel_size_nm': float(voxel_size_nm),
        'n_px': int(g.i_end - g.i_start),
        'x_edge0_nm': float(g.voxel_centers[g.i_start] - voxel_size_nm / 2.0),
        'y_edge_top_nm': float(g.voxel_centers[g.i_end - 1] + voxel_size_nm / 2.0),
        'z_slices_nm': [float(v) for v in z],
    }


def with_focal_plane(geom, z_nm):
    """Copy of geom with a single focal plane at depth z_nm."""
    return dict(geom, z_slices_nm=[float(z_nm)])


def to_pixels(xyz_nm, geom):
    """nm -> continuous pixel coords (x_px, y_px), nearest slice, dz."""
    v = geom['voxel_size_nm']
    x_px = (xyz_nm[..., 0] - geom['x_edge0_nm']) / v
    y_px = (geom['y_edge_top_nm'] - xyz_nm[..., 1]) / v
    zs = np.asarray(geom['z_slices_nm'])
    if len(zs):
        dz_all = xyz_nm[..., 2][..., None] - zs
        k = np.argmin(np.abs(dz_all), axis=-1)
        dz = np.take_along_axis(dz_all, k[..., None], axis=-1)[..., 0]
    else:
        k = np.full(xyz_nm.shape[:-1], -1)
        dz = np.full(xyz_nm.shape[:-1], np.nan)
    return x_px, y_px, k, dz
