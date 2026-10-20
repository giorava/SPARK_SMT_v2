"""
exposure.py - What the camera integrates in each frame.

run_full_simulation() stores, for every display frame, the positions of all
molecules at every physics sub-step of the exposure window
('pos_steps', (steps, N, 3) in µm), their modal state ('modal_states'),
the mean site positions ('site_centroid') and which sites were occupied at
any point of the window ('any_occupied').
"""

import pandas as pd


def compute_exposure_frames(exposure_data):
    """Per-frame summaries of exposure_data.

    Returns
    -------
    mol_data  : {frame: DataFrame(trajectory, x, y, z, state, frame)} with
                positions averaged over the exposure window (µm)
    site_data : {frame: (site_centroid (S, 3) µm, any_occupied (S,) bool)}
    """
    mol_data = {}
    site_data = {}
    for frame, exp in exposure_data.items():
        if exp is None:
            continue
        pos_steps = exp['pos_steps']          # (steps, N, 3)
        n = pos_steps.shape[1]
        centroid_pos = pos_steps.mean(axis=0) # (N, 3)
        mol_data[frame] = pd.DataFrame({
            'trajectory': range(n),
            'x': centroid_pos[:, 0], 'y': centroid_pos[:, 1], 'z': centroid_pos[:, 2],
            'state': exp['modal_states'], 'frame': frame})
        site_data[frame] = (exp['site_centroid'], exp['any_occupied'])
    return mol_data, site_data
