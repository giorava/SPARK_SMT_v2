"""
site_motion.py - Chromatin (binding-site cluster) motion.

Each cluster moves by fractional Brownian motion (fBm) with its own
diffusion coefficient and a common Hurst exponent. Steps are generated one
display frame at a time with the Davies-Harte method and conditioned on the
last step of the previous frame, so the increments are an exact fBm across
frame boundaries.
"""

import numpy as np


def generate_all_fbm_steps_dh(total_steps, num_clusters, cluster_D, hurst, sim_dt_s,
                              rng=None):
    if rng is None:
        rng = np.random
    n = int(total_steps)
    H = float(np.clip(hurst, 0.01, 0.99))
    step_stds = np.sqrt(2.0 * np.maximum(cluster_D, 0.0) * sim_dt_s)
    if n < 2 or num_clusters == 0:
        out = np.zeros((n, num_clusters, 3))
        for d in range(3):
            out[:, :, d] = rng.normal(size=(n, num_clusters)) * step_stds[np.newaxis, :]
        return out
    N = 2 * n
    k = np.arange(n + 1, dtype=float)
    autocov = 0.5 * (np.abs(k - 1) ** (2.0 * H) - 2.0 * np.abs(k) ** (2.0 * H) + np.abs(k + 1) ** (2.0 * H))
    r = np.empty(N)
    r[:n + 1] = autocov
    r[n + 1:] = autocov[1:n][::-1]
    lam = np.fft.fft(r).real
    lam = np.maximum(lam, 0.0)
    n_series = num_clusters * 3
    w = np.zeros((n_series, N), dtype=complex)
    w[:, 0] = np.sqrt(lam[0] / N) * rng.normal(size=n_series)
    w[:, n] = np.sqrt(lam[n] / N) * rng.normal(size=n_series)
    scale_mid = np.sqrt(lam[1:n] / (2.0 * N))
    w[:, 1:n] = scale_mid[np.newaxis, :] * (
        rng.normal(size=(n_series, n - 1)) + 1j * rng.normal(size=(n_series, n - 1)))
    w[:, n + 1:] = np.conj(w[:, 1:n][:, ::-1])
    fgn = np.fft.fft(w, axis=1).real[:, :n]
    fgn = fgn.reshape(num_clusters, 3, n).transpose(2, 0, 1)
    steps = fgn * step_stds[np.newaxis, :, np.newaxis]
    return steps


def generate_fbm_step(prev_step, hurst, num_clusters, cluster_D, sim_dt_s,
                      rng=None):
    if rng is None:
        rng = np.random
    step_std = np.sqrt(2 * cluster_D * sim_dt_s)
    new_random = rng.normal(0, 1, (num_clusters, 3))
    rho = 2.0 ** (2.0 * hurst - 1.0) - 1.0
    rho = np.clip(rho, -0.99, 0.99)
    if prev_step is not None:
        prev_norms = np.linalg.norm(prev_step, axis=1, keepdims=True)
        prev_norms = np.where(prev_norms > 1e-10, prev_norms, 1.0)
        prev_direction = prev_step / prev_norms
        combined = rho * prev_direction + np.sqrt(1.0 - rho ** 2) * new_random
    else:
        combined = new_random
    step = combined * step_std[:, np.newaxis]
    return step


def frame_fbm_steps(n_steps, num_clusters, cluster_D, hurst, sim_dt_s,
                    prev_step=None, rng=None):
    """fBm cluster steps for one display frame, shape (n_steps, K, 3).

    Generates the block with Davies-Harte, then, if `prev_step` (the last
    step of the previous frame) is given, conditions it so the first step
    follows prev_step with the exact lag-1 fGn correlation (Gaussian
    bridge) and the remaining steps are adjusted by their lag-k
    autocovariance. This keeps the motion an exact fBm across frames.
    """
    H = float(hurst)
    step_stds = np.sqrt(2.0 * np.maximum(cluster_D, 0.0) * sim_dt_s)
    r1 = 0.5 * (2.0 ** (2.0 * H) - 2.0)  # fGn autocovariance at lag 1
    dh = generate_all_fbm_steps_dh(
        n_steps, num_clusters, cluster_D, hurst, sim_dt_s, rng=rng)
    if prev_step is not None:
        # Exact Gaussian bridge: r(1)*prev + sqrt(1-r(1)^2)*std*Z
        bridge = (r1 * prev_step
                  + np.sqrt(1.0 - r1 ** 2)
                  * step_stds[:, np.newaxis]
                  * rng.normal(size=prev_step.shape))
        # Condition DH block on first element = bridge
        k = np.arange(1, n_steps, dtype=float)
        rk = 0.5 * (np.abs(k - 1) ** (2 * H)
                    - 2 * np.abs(k) ** (2 * H)
                    + np.abs(k + 1) ** (2 * H))
        dh[1:] += rk[:, np.newaxis, np.newaxis] * (
            bridge[np.newaxis, :, :] - dh[0:1, :, :])
        dh[0] = bridge
    return dh
