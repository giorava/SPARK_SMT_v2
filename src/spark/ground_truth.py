"""
ground_truth.py - Ground-truth renderings: molecules coloured by state and
binding sites drawn as circles, one RGB image per z-slice.
"""

import io

import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from .geometry import view_grid, slice_z_positions


def render_ground_truth(frame_df, nucleus_diameter_nm,
                                  molecule_diameter_nm, image_dpi, slice_spacing_nm,
                                  binding_sites_pos, site_occupied,
                                  site_strengths=None, strength_range=(0.1, 10.0),
                                  is_synthetic=None, draw_boundary=False,
                                  upscale_factor=12, voxel_size_nm=108.0,
                                  num_slices=None):
    """One RGB PIL image per z-slice for one frame.

    Molecules (exposure-averaged positions, modal state) within half a
    molecule diameter of a slice are drawn coloured by state (cyan free,
    yellow intermediate, red bound, magenta non-specific). Sites within
    50 nm of the slice are circles: green = synthetic, otherwise shaded by
    strength; filled when occupied during the exposure.
    """
    NUCLEUS_RADIUS_NM = nucleus_diameter_nm / 2
    coords = frame_df[['x', 'y', 'z']].values * 1000
    plot_limit = NUCLEUS_RADIUS_NM * 1.1
    g = view_grid(nucleus_diameter_nm, voxel_size_nm)
    voxel_centers_1d, i_start, i_end = g.voxel_centers, g.i_start, g.i_end
    convolved_native_pixels = i_end - i_start
    target_final_pixels = convolved_native_pixels * upscale_factor
    native_pixels_gt = max(target_final_pixels, int(np.ceil(2 * plot_limit / molecule_diameter_nm)))
    crop_min = voxel_centers_1d[i_start] - voxel_size_nm / 2
    crop_max = voxel_centers_1d[i_end - 1] + voxel_size_nm / 2
    images = []
    all_z_positions = slice_z_positions(NUCLEUS_RADIUS_NM, slice_spacing_nm, num_slices)
    log_min = np.log10(max(strength_range[0], 1e-6))
    log_max = np.log10(max(strength_range[1], 1e-6))

    for z_nm in all_z_positions:
        native_fig_size = native_pixels_gt / image_dpi
        fig_gt, ax_gt = plt.subplots(figsize=(native_fig_size, native_fig_size), facecolor='black')
        ax_gt.set_position([0, 0, 1, 1])
        ax_gt.set_axis_off()
        nuc_cross_section_r = np.sqrt(max(0, NUCLEUS_RADIUS_NM ** 2 - z_nm ** 2))
        if draw_boundary:
            ax_gt.add_patch(plt.Circle((0, 0), nuc_cross_section_r,
                                       fill=False, edgecolor='cyan', ls='--', lw=1.5))
        mol_rad = molecule_diameter_nm / 2
        dists = np.abs(coords[:, 2] - z_nm)
        m_slice = dists < mol_rad
        if np.any(m_slice):
            c_map = {0: 'cyan', 1: 'yellow', 2: 'red', 3: 'magenta'}
            state_vals = frame_df['state'].values[m_slice].astype(int)
            colors = [c_map.get(s, 'white') for s in state_vals]
            data_width_nm = crop_max - crop_min
            fig_width_points = native_fig_size * 72
            points_per_nm = fig_width_points / data_width_nm
            marker_size_points = molecule_diameter_nm * points_per_nm
            marker_area = marker_size_points ** 2
            ax_gt.scatter(coords[m_slice, 0], coords[m_slice, 1],
                          c=colors, s=marker_area, linewidths=0)
        if binding_sites_pos is not None:
            s_coords = binding_sites_pos * 1000
            s_slice = np.abs(s_coords[:, 2] - z_nm) < 50
            slice_indices = np.where(s_slice)[0]
            for idx in slice_indices:
                s = s_coords[idx]
                r = np.sqrt(max(0, 50 ** 2 - (s[2] - z_nm) ** 2))
                is_synth = (is_synthetic is not None and is_synthetic[idx])
                if is_synth:
                    edge_color = (0.2, 1.0, 0.2)
                else:
                    if site_strengths is not None and idx < len(site_strengths):
                        strength = site_strengths[idx]
                        log_strength = np.log10(np.clip(strength, strength_range[0], strength_range[1]))
                        norm_strength = (log_strength - log_min) / (log_max - log_min) if log_max > log_min else 0.5
                    else:
                        norm_strength = 0.5
                    r_channel = 1.0 - 0.4 * norm_strength
                    g_channel = 0.7 * (1 - norm_strength)
                    b_channel = 0.7 * (1 - norm_strength)
                    edge_color = (r_channel, g_channel, b_channel)
                is_occupied = (site_occupied is not None and site_occupied[idx])
                if is_occupied:
                    ax_gt.add_patch(plt.Circle((s[0], s[1]), r, fill=True,
                                               facecolor=(*edge_color, 0.6), edgecolor=edge_color, lw=1.5))
                else:
                    ax_gt.add_patch(plt.Circle((s[0], s[1]), r, fill=False,
                                               edgecolor=edge_color, lw=1.5))
        ax_gt.set_xlim(crop_min, crop_max)
        ax_gt.set_ylim(crop_min, crop_max)
        ax_gt.set_aspect('equal')
        buf = io.BytesIO()
        plt.savefig(buf, dpi=image_dpi, facecolor='black', pad_inches=0, format='png')
        buf.seek(0)
        native_img = Image.open(buf).convert('RGB')
        plt.close()
        images.append(native_img.resize((target_final_pixels, target_final_pixels),
                                        Image.LANCZOS))
    return images
