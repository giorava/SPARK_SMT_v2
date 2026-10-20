"""
smt_tracking.py - Binding-event recording and single-molecule-tracking (SMT)
image simulation for SPARK.

WHAT IT DOES
------------
1. BindingTracker is handed to run_full_simulation(tracker=...). At every
   physics step it compares each molecule's (state, bound site, unbind time)
   with the previous step and logs ONLY the changes. From that event log the
   exact state and bound site of EVERY molecule at EVERY step can be rebuilt,
   so nothing is lost, but storage stays small.

2. One or more molecules are "labelled". Labelling is pure bookkeeping: all
   molecules are simulated identically, so the label can even be chosen AFTER
   the simulation (e.g. "a molecule that visited the synthetic array").

3. SMT images are rendered from the labelled molecules only. Unlabelled
   molecules still diffuse, bind and compete for sites, but emit no light.
   The PSF is placed at the exact sub-pixel position of the molecule at every
   physics sub-step of the exposure (no voxel snapping), so motion blur and
   localization ground truth are exact.

STEP / TIME CONVENTION
----------------------
Step k is the snapshot taken BEFORE physics step k runs, at time k * sim_dt.
This is the same snapshot the engine stores in its exposure buffers, so:
    frame       = k // steps_per_display
    in_exposure = (k % steps_per_display) < steps_per_exposure
An event logged at step k first becomes visible in snapshot k (it happened
during physics step k-1). A molecule bound from bind_step to unbind_step is
seen bound in snapshots bind_step .. unbind_step-1, so dwell = (unbind_step -
bind_step) * sim_dt.

STATE CODES:  0 free, 1 intermediate, 2 specifically bound, 3 non-specific.
SITE IDS:     index into the engine's site array. Synthetic sites are the
              LAST num_synthetic_sites entries; synthetic_index 0..n-1 gives
              their order within the array.

PIXEL CONVENTION
----------------
x_px / y_px are continuous coordinates in the saved (non-upscaled) image:
pixel (column c, row r) covers [c, c+1) x [r, r+1), so its centre is at
(c + 0.5, r + 0.5). Row 0 is the top of the image (largest y). This matches
the convolved images SPARK already writes.
"""

import json
import os

import numpy as np
import pandas as pd
from scipy.special import erf
from scipy.spatial import cKDTree

STATE_LABELS = {0: 'free', 1: 'intermediate', 2: 'bound', 3: 'nonspecific'}
_SQRT2 = np.sqrt(2.0)
_FWHM_TO_SIGMA = 1.0 / (2.0 * np.sqrt(2.0 * np.log(2.0)))


# ============================================================================
# LABEL SPECIFICATION
# ============================================================================

def parse_label_spec(spec):
    """Parse a label spec string.

    Accepted forms (case-insensitive):
        "random:N"     N molecules chosen at random (default "random:1")
        "ids:3,17"     explicit molecule ids (also "3,17" or "3;17" or "3")
        "synthetic:N"  N molecules chosen at random among those that were
                       bound to the synthetic array at least once
    Returns (mode, value) with mode in {'random', 'ids', 'synthetic'}.
    """
    if spec is None or (isinstance(spec, float) and np.isnan(spec)):
        return 'random', 1
    s = str(spec).strip().lower()
    if not s:
        return 'random', 1
    if ':' in s:
        mode, val = s.split(':', 1)
        mode, val = mode.strip(), val.strip()
    else:
        mode, val = 'ids', s
    if mode in ('random', 'synthetic'):
        n = int(float(val)) if val else 1
        if n < 1:
            raise ValueError("Label spec %r: N must be >= 1" % spec)
        return mode, n
    if mode in ('ids', 'id'):
        parts = [p for p in val.replace(';', ',').replace(' ', ',').split(',') if p]
        ids = sorted(set(int(float(p)) for p in parts))
        if not ids:
            raise ValueError("Label spec %r has no ids" % spec)
        return 'ids', ids
    raise ValueError("Unknown label spec %r (use random:N, ids:a,b or "
                     "synthetic:N)" % spec)


def make_label_rng(seed):
    """RNG used only for choosing labels / SMT camera noise.

    Kept separate from the simulation RNGs so labelling never changes the
    physics. seed=None gives a non-reproducible choice.
    """
    if seed is None:
        return np.random.default_rng()
    return np.random.default_rng(np.random.SeedSequence([int(seed), 0x5317]))


# ============================================================================
# TRACKER (called from inside run_full_simulation)
# ============================================================================

class BindingTracker(object):
    """Records every state / binding change of every molecule.

    Parameters
    ----------
    label_spec : str
        Which molecules to label (see parse_label_spec). For 'random' and
        'ids' specs the labelled molecules are known before the run, so their
        positions are recorded at EVERY physics step. For 'synthetic' the
        choice is made afterwards and positions come from the exposure
        windows (what the camera sees) or the trajectory dump.
    trajectory_stride : int
        If > 0, also store positions / states / bound site of ALL molecules
        every `trajectory_stride` steps (can be large: N x 3 float32 per
        stored step).
    label_seed : int or None
        Seed for random label choice.
    """

    def __init__(self, label_spec='random:1', trajectory_stride=0,
                 label_seed=None):
        self.label_spec = label_spec
        self.label_mode, self.label_value = parse_label_spec(label_spec)
        self.trajectory_stride = int(trajectory_stride or 0)
        self.label_seed = label_seed
        self._started = False

    # -- engine hooks -----------------------------------------------------
    def start(self, num_molecules, is_synthetic, site_strengths, sim_dt_s,
              steps_per_display, steps_per_exposure, num_display_frames):
        self.num_molecules = int(num_molecules)
        self.is_synthetic = np.asarray(is_synthetic, dtype=bool)
        self.site_strengths = np.asarray(site_strengths, dtype=float)
        self.synthetic_site_ids = np.where(self.is_synthetic)[0].astype(np.int32)
        self.sim_dt_s = float(sim_dt_s)
        self.steps_per_display = int(steps_per_display)
        self.steps_per_exposure = int(steps_per_exposure)
        self.num_display_frames = int(num_display_frames)

        # Labels that can be fixed before the run get full per-step tracks.
        rng = make_label_rng(self.label_seed)
        if self.label_mode == 'random':
            n = min(self.label_value, self.num_molecules)
            self.watch_ids = np.sort(rng.choice(self.num_molecules, n,
                                                replace=False)).astype(np.int64)
        elif self.label_mode == 'ids':
            ids = np.asarray(self.label_value, dtype=np.int64)
            bad = ids[(ids < 0) | (ids >= self.num_molecules)]
            if len(bad):
                raise ValueError("Labelled ids %s out of range: this run has "
                                 "%d molecules (ids 0..%d)" % (
                                     bad.tolist(), self.num_molecules,
                                     self.num_molecules - 1))
            self.watch_ids = ids
        else:
            self.watch_ids = np.zeros(0, dtype=np.int64)

        self._ev = {k: [] for k in ('step', 'mol', 'fs', 'ts', 'fsite', 'tsite')}
        self._watch_pos = []
        self._synth_pos = []
        self._traj = {'step': [], 'pos': [], 'states': [], 'bound': []}
        self._prev = None
        self._last_step = -1
        self._started = True

    def observe(self, step, pos, states, sites, bound_site_indices,
                unbind_times):
        if not self._started:
            raise RuntimeError("BindingTracker.observe called before start")
        step = int(step)
        if step == self._last_step:
            return
        self._last_step = step
        states = np.asarray(states)
        bound = np.asarray(bound_site_indices)

        if self._prev is None:
            # Initial condition: one 'init' row per molecule.
            n = len(states)
            self._append(np.full(n, step), np.arange(n), np.full(n, -1),
                         states, np.full(n, -1), bound)
        else:
            ps, pb, pu = self._prev
            changed = (states != ps) | (bound != pb) | (
                (bound >= 0) & (pb >= 0) & (unbind_times != pu))
            idx = np.nonzero(changed)[0]
            if len(idx):
                self._append(np.full(len(idx), step), idx, ps[idx],
                             states[idx], pb[idx], bound[idx])
        self._prev = (states.copy(), bound.copy(),
                      np.asarray(unbind_times).copy())

        if len(self.watch_ids):
            self._watch_pos.append(pos[self.watch_ids].astype(np.float32))
        if len(self.synthetic_site_ids):
            self._synth_pos.append(sites[self.synthetic_site_ids].astype(np.float32))
        if self.trajectory_stride > 0 and step % self.trajectory_stride == 0:
            self._traj['step'].append(step)
            self._traj['pos'].append(pos.astype(np.float32))
            self._traj['states'].append(states.astype(np.int8))
            self._traj['bound'].append(bound.astype(np.int32))

    def _append(self, step, mol, fs, ts, fsite, tsite):
        self._ev['step'].append(np.asarray(step, dtype=np.int32))
        self._ev['mol'].append(np.asarray(mol, dtype=np.int32))
        self._ev['fs'].append(np.asarray(fs, dtype=np.int8))
        self._ev['ts'].append(np.asarray(ts, dtype=np.int8))
        self._ev['fsite'].append(np.asarray(fsite, dtype=np.int32))
        self._ev['tsite'].append(np.asarray(tsite, dtype=np.int32))

    # -- export -----------------------------------------------------------
    def result(self):
        """Plain dict of numpy arrays (picklable, safe to cache)."""
        if not self._started:
            raise RuntimeError("Tracker never started (simulation not run?)")
        cat = lambda key, dt: (np.concatenate(self._ev[key]) if self._ev[key]
                               else np.zeros(0, dtype=dt))
        n_syn = len(self.synthetic_site_ids)
        T1 = self._last_step + 1
        res = {
            'version': 1,
            'label_spec': self.label_spec,
            'label_seed': self.label_seed,
            'num_molecules': self.num_molecules,
            'num_sites': len(self.is_synthetic),
            'is_synthetic': self.is_synthetic,
            'site_strengths': self.site_strengths,
            'synthetic_site_ids': self.synthetic_site_ids,
            'sim_dt_s': self.sim_dt_s,
            'final_step': self._last_step,
            'steps_per_display': self.steps_per_display,
            'steps_per_exposure': self.steps_per_exposure,
            'num_display_frames': self.num_display_frames,
            'ev_step': cat('step', np.int32), 'ev_mol': cat('mol', np.int32),
            'ev_from_state': cat('fs', np.int8), 'ev_to_state': cat('ts', np.int8),
            'ev_from_site': cat('fsite', np.int32),
            'ev_to_site': cat('tsite', np.int32),
            'watch_ids': self.watch_ids,
            'watch_pos': (np.stack(self._watch_pos) if self._watch_pos
                          else np.zeros((T1, 0, 3), np.float32)),
            'synth_pos': (np.stack(self._synth_pos) if self._synth_pos
                          else np.zeros((T1, n_syn, 3), np.float32)),
        }
        if self._traj['step']:
            res['traj_step'] = np.asarray(self._traj['step'], dtype=np.int32)
            res['traj_pos'] = np.stack(self._traj['pos'])
            res['traj_states'] = np.stack(self._traj['states'])
            res['traj_bound'] = np.stack(self._traj['bound'])
        return res


# ============================================================================
# EVENT TABLES
# ============================================================================

def events_dataframe(res, labeled_ids=()):
    """Every change of every molecule, one row per (step, molecule)."""
    fs, ts = res['ev_from_site'], res['ev_to_site']
    is_syn = res['is_synthetic']
    syn_index = np.full(len(is_syn), -1, dtype=np.int32)
    syn_index[res['synthetic_site_ids']] = np.arange(len(res['synthetic_site_ids']))

    event = np.full(len(fs), 'state', dtype=object)
    event[(fs < 0) & (ts >= 0)] = 'bind'
    event[(fs >= 0) & (ts < 0)] = 'unbind'
    event[(fs >= 0) & (ts >= 0)] = 'rebind'   # unbind + bind in one step
    event[res['ev_from_state'] < 0] = 'init'

    def _syn(site):
        out = np.zeros(len(site), dtype=bool)
        m = site >= 0
        out[m] = is_syn[site[m]]
        return out

    df = pd.DataFrame({
        'step': res['ev_step'],
        'time_s': res['ev_step'] * res['sim_dt_s'],
        'molecule_id': res['ev_mol'],
        'event': event,
        'from_state': res['ev_from_state'],
        'to_state': res['ev_to_state'],
        'from_site': fs,
        'to_site': ts,
        'from_site_synthetic': _syn(fs),
        'to_site_synthetic': _syn(ts),
        'to_synthetic_index': np.where(ts >= 0, syn_index[np.maximum(ts, 0)], -1),
    })
    df['labeled'] = df['molecule_id'].isin(np.asarray(labeled_ids, dtype=int))
    return df


def dwell_table(res, labeled_ids=()):
    """One row per binding interval (molecule on a site)."""
    step, mol = res['ev_step'], res['ev_mol']
    fs, ts = res['ev_from_site'], res['ev_to_site']
    b = pd.DataFrame({'molecule_id': mol[ts >= 0], 'site_id': ts[ts >= 0],
                      'bind_step': step[ts >= 0]})
    u = pd.DataFrame({'molecule_id': mol[fs >= 0], 'site_u': fs[fs >= 0],
                      'unbind_step': step[fs >= 0]})
    b = b.sort_values(['molecule_id', 'bind_step'], kind='stable')
    u = u.sort_values(['molecule_id', 'unbind_step'], kind='stable')
    b['k'] = b.groupby('molecule_id').cumcount()
    u['k'] = u.groupby('molecule_id').cumcount()
    d = b.merge(u, on=['molecule_id', 'k'], how='left')
    mismatch = d['site_u'].notna() & (d['site_u'] != d['site_id'])
    if mismatch.any():
        raise RuntimeError("Internal error: bind/unbind sites do not pair up "
                           "(%d rows)" % int(mismatch.sum()))
    d['censored'] = d['unbind_step'].isna()
    d['unbind_step'] = d['unbind_step'].fillna(res['final_step']).astype(np.int64)
    dt = res['sim_dt_s']
    is_syn = res['is_synthetic']
    syn_index = np.full(len(is_syn), -1, dtype=np.int64)
    syn_index[res['synthetic_site_ids']] = np.arange(len(res['synthetic_site_ids']))
    site = d['site_id'].values.astype(np.int64)
    out = pd.DataFrame({
        'molecule_id': d['molecule_id'].values,
        'site_id': site,
        'is_synthetic': is_syn[site],
        'synthetic_index': syn_index[site],
        'site_strength': res['site_strengths'][site],
        'bind_step': d['bind_step'].values,
        'unbind_step': d['unbind_step'].values,
        'bind_time_s': d['bind_step'].values * dt,
        'unbind_time_s': d['unbind_step'].values * dt,
        'dwell_s': (d['unbind_step'].values - d['bind_step'].values) * dt,
        'censored': d['censored'].values,
    })
    # Flag intervals that start the very step the same molecule left the
    # same site (unbind + immediate recapture in one physics step).
    out = out.sort_values(['molecule_id', 'bind_step'], kind='stable')
    prev_site = out.groupby('molecule_id')['site_id'].shift()
    prev_end = out.groupby('molecule_id')['unbind_step'].shift()
    out['immediate_rebind_same_site'] = ((prev_site == out['site_id'])
                                         & (prev_end == out['bind_step'])).values
    out['labeled'] = out['molecule_id'].isin(np.asarray(labeled_ids, dtype=int))
    return out.sort_values(['bind_step', 'molecule_id']).reset_index(drop=True)


def merge_visits(dwells, max_gap_steps=0, sim_dt_s=1.0):
    """Merge consecutive intervals of one molecule on the synthetic array.

    Intervals are merged when the gap between them is <= max_gap_steps
    (0 = only back-to-back intervals, i.e. instant recapture). Any synthetic
    site counts, so hopping between sites of the array is one visit.
    """
    d = dwells[dwells['is_synthetic']].sort_values(['molecule_id', 'bind_step'])
    rows = []
    for mol, g in d.groupby('molecule_id', sort=False):
        cur = None
        for r in g.itertuples(index=False):
            if cur is not None and r.bind_step - cur['unbind_step'] <= max_gap_steps:
                cur['unbind_step'] = max(cur['unbind_step'], r.unbind_step)
                cur['censored'] = r.censored
                cur['n_intervals'] += 1
                cur['sites'].add(int(r.synthetic_index))
            else:
                if cur is not None:
                    rows.append(cur)
                cur = {'molecule_id': int(mol), 'bind_step': int(r.bind_step),
                       'unbind_step': int(r.unbind_step), 'censored': bool(r.censored),
                       'n_intervals': 1, 'sites': {int(r.synthetic_index)},
                       'labeled': bool(r.labeled)}
        if cur is not None:
            rows.append(cur)
    out = pd.DataFrame(rows, columns=['molecule_id', 'bind_step', 'unbind_step',
                                      'censored', 'n_intervals', 'sites', 'labeled'])
    out['synthetic_indices'] = out['sites'].apply(
        lambda s: ';'.join(str(i) for i in sorted(s)))
    out = out.drop(columns='sites')
    out['bind_time_s'] = out['bind_step'] * sim_dt_s
    out['unbind_time_s'] = out['unbind_step'] * sim_dt_s
    out['visit_s'] = (out['unbind_step'] - out['bind_step']) * sim_dt_s
    return out.sort_values(['bind_step', 'molecule_id']).reset_index(drop=True)


def resolve_labels(res, label_spec, rng, dwells=None):
    """Turn a label spec into molecule ids. Returns (ids, note)."""
    mode, val = parse_label_spec(label_spec)
    n_mol = res['num_molecules']
    if mode == 'ids':
        ids = np.asarray(val, dtype=np.int64)
        bad = ids[(ids < 0) | (ids >= n_mol)]
        if len(bad):
            raise ValueError("Labelled ids %s out of range (0..%d)" % (
                bad.tolist(), n_mol - 1))
        return ids, ''
    if mode == 'random':
        if res['label_spec'] == label_spec and len(res['watch_ids']):
            return np.asarray(res['watch_ids'], dtype=np.int64), ''
        n = min(val, n_mol)
        return np.sort(rng.choice(n_mol, n, replace=False)).astype(np.int64), ''
    # mode == 'synthetic'
    if len(res['synthetic_site_ids']) == 0:
        raise ValueError("Label spec %r needs num_synthetic_sites > 0" % label_spec)
    if dwells is None:
        dwells = dwell_table(res)
    binders = np.unique(dwells.loc[dwells['is_synthetic'], 'molecule_id'].values)
    if len(binders) == 0:
        n = min(val, n_mol)
        ids = np.sort(rng.choice(n_mol, n, replace=False)).astype(np.int64)
        return ids, ("No molecule bound the synthetic array; fell back to %d "
                     "random molecule(s)." % n)
    note = ''
    if len(binders) < val:
        note = ("Only %d molecule(s) bound the synthetic array; labelling all "
                "of them." % len(binders))
    n = min(val, len(binders))
    return np.sort(rng.choice(binders, n, replace=False)).astype(np.int64), note


def state_timeline(res, mol_ids, n_steps):
    """(n_steps, L) arrays of state and bound site, rebuilt from events."""
    mol_ids = np.asarray(mol_ids, dtype=np.int64)
    steps = np.arange(n_steps)
    st = np.zeros((n_steps, len(mol_ids)), dtype=np.int8)
    bs = np.full((n_steps, len(mol_ids)), -1, dtype=np.int32)
    order = np.lexsort((res['ev_step'], res['ev_mol']))
    em, es = res['ev_mol'][order], res['ev_step'][order]
    ets, etsite = res['ev_to_state'][order], res['ev_to_site'][order]
    for j, m in enumerate(mol_ids):
        lo, hi = np.searchsorted(em, m, 'left'), np.searchsorted(em, m, 'right')
        k = np.searchsorted(es[lo:hi], steps, 'right') - 1 + lo
        st[:, j] = ets[k]
        bs[:, j] = etsite[k]
    return st, bs


# ============================================================================
# IMAGE GEOMETRY (mirrors spark.geometry's crop / slice logic)
# ============================================================================

def image_geometry(nucleus_diameter_nm, voxel_size_nm, slice_spacing_nm,
                   num_slices=None):
    R = nucleus_diameter_nm / 2.0
    plot_limit = R * 1.1
    G = int(np.ceil(nucleus_diameter_nm / voxel_size_nm)) + 2
    if G % 2 == 0:
        G += 1
    half = G * voxel_size_nm / 2.0
    centers = np.arange(G) * voxel_size_nm - half + voxel_size_nm / 2.0
    view = np.where((centers >= -plot_limit) & (centers <= plot_limit))[0]
    i0, i1 = (view[0], view[-1] + 1) if len(view) else (0, G)
    if num_slices is not None:
        nh = int(num_slices) // 2
        cand = sorted({i * slice_spacing_nm for i in range(-nh, nh + 1)})
        z = [zz for zz in cand if abs(zz) <= R]
    else:
        z = sorted(np.unique(np.concatenate([
            np.arange(0, R, slice_spacing_nm),
            -np.arange(0, R, slice_spacing_nm)])))
    return {
        'voxel_size_nm': float(voxel_size_nm),
        'n_px': int(i1 - i0),
        'x_edge0_nm': float(centers[i0] - voxel_size_nm / 2.0),
        'y_edge_top_nm': float(centers[i1 - 1] + voxel_size_nm / 2.0),
        'z_slices_nm': [float(v) for v in z],
    }


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


# ============================================================================
# SMT RENDERING
# ============================================================================

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


def array_centroid_per_frame(res, num_frames, spd, spe):
    """(F, 3) synthetic-array centroid in nm, averaged over all synthetic
    sites and over the exposure window of each frame. None if no array."""
    if len(res['synthetic_site_ids']) == 0:
        return None
    last = len(res['synth_pos']) - 1
    out = np.empty((int(num_frames), 3))
    for f in range(int(num_frames)):
        k0 = min(f * spd, last)
        k1 = min(k0 + max(int(spe), 1), last + 1)
        out[f] = res['synth_pos'][k0:k1].reshape(-1, 3).mean(axis=0) * 1000.0
    return out


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


# ============================================================================
# POSITIONS OF LABELLED MOLECULES
# ============================================================================

def _exposure_window_positions(exposure_data, frame, mol_ids):
    """(S, L, 3) positions in um during the exposure of `frame`, or None."""
    exp = exposure_data.get(frame) if exposure_data is not None else None
    if exp is None:
        return None
    if isinstance(exp, dict):
        return np.asarray(exp['pos_steps'])[:, mol_ids, :]
    if len(exp) == 0:
        return None
    return np.array([s['pos'][mol_ids] for s in exp])   # rebinned format


def labeled_positions(res, exposure_data, mol_ids, n_steps, spd, spe):
    """(n_steps, L, 3) positions in nm; NaN where not recorded."""
    mol_ids = np.asarray(mol_ids, dtype=np.int64)
    out = np.full((n_steps, len(mol_ids), 3), np.nan)
    # 1) exposure windows (always available: this is what the camera saw)
    for f in range(n_steps // spd + 1):
        P = _exposure_window_positions(exposure_data, f, mol_ids)
        if P is None:
            continue
        k0 = f * spd
        m = min(len(P), n_steps - k0)
        if m > 0:
            out[k0:k0 + m] = P[:m] * 1000.0
    # 2) full trajectory dump (every `stride` steps, all molecules)
    if 'traj_step' in res:
        ok = res['traj_step'] < n_steps
        out[res['traj_step'][ok]] = res['traj_pos'][ok][:, mol_ids, :] * 1000.0
    # 3) watched molecules: every step
    w = list(res['watch_ids'])
    if w:
        for j, m in enumerate(mol_ids):
            if m in w:
                out[:, j, :] = res['watch_pos'][:n_steps, w.index(m), :] * 1000.0
    return out


# ============================================================================
# MAIN OUTPUT WRITER
# ============================================================================

def write_tracking_outputs(row_outdir, res, exposure_data, label_spec,
                           label_rng, geom, num_frames, spd, spe,
                           lateral_fwhm_nm, axial_fwhm_nm,
                           near_radius_nm=250.0, write_smt=False,
                           smt_params=None, bit_depth=16, log=print):
    """Write binding tables, labelled tracks and (optionally) SMT images.

    Returns the list of labelled molecule ids.
    """
    tdir = os.path.join(row_outdir, 'Tracking')
    os.makedirs(tdir, exist_ok=True)
    dt = res['sim_dt_s']
    n_steps = int(num_frames) * int(spd)
    n_steps = min(n_steps, res['final_step'])

    # ---- focal plane: optionally a single plane at the synthetic array ---
    n_syn0 = len(res['synthetic_site_ids'])
    focus_mode = (smt_params or {}).get('focus', 'slices') if write_smt else 'slices'
    focus_z = None   # per-frame focal depth (nm) in 'track' mode
    if focus_mode == 'track':
        if n_syn0:
            # Focus lock on the array: each frame's single focal plane sits at
            # the array's mean depth during THAT frame's exposure window.
            last = len(res['synth_pos']) - 1
            focus_z = np.empty(int(num_frames))
            for f in range(int(num_frames)):
                k0 = min(f * spd, last)
                k1 = min(k0 + max(int(spe), 1), last + 1)
                focus_z[f] = res['synth_pos'][k0:k1, :, 2].mean() * 1000.0
            geom = dict(geom, z_slices_nm=[float(focus_z[0])])
            log("  Tracking: SMT focus follows the synthetic array, z = %.0f "
                "to %.0f nm over the movie" % (focus_z.min(), focus_z.max()))
        else:
            log("  Tracking: --smt-focus track ignored (no synthetic sites)")
    elif focus_mode == 'synthetic':
        if n_syn0:
            z_focus = float(res['synth_pos'][0, :, 2].mean() * 1000.0)
            geom = dict(geom, z_slices_nm=[z_focus])
            log("  Tracking: SMT focal plane set to synthetic array depth "
                "z = %.0f nm (fixed; the array may drift afterwards)" % z_focus)
        else:
            log("  Tracking: --smt-focus synthetic ignored (no synthetic sites)")

    # ---- events & dwells (all molecules) --------------------------------
    dwells = dwell_table(res)
    labeled, note = resolve_labels(res, label_spec, label_rng, dwells)
    if note:
        log("  Tracking: " + note)
    dwells['labeled'] = dwells['molecule_id'].isin(labeled)
    ev = events_dataframe(res, labeled)
    ev.to_csv(os.path.join(tdir, 'state_events.csv'), index=False)
    dwells.to_csv(os.path.join(tdir, 'site_dwells.csv'), index=False)
    syn_dw = dwells[dwells['is_synthetic']]
    syn_dw.to_csv(os.path.join(tdir, 'synthetic_dwells.csv'), index=False)
    visits = merge_visits(dwells, max_gap_steps=0, sim_dt_s=dt)
    visits.to_csv(os.path.join(tdir, 'synthetic_visits.csv'), index=False)

    # ---- synthetic occupancy per step ------------------------------------
    n_syn = len(res['synthetic_site_ids'])
    T1 = res['final_step'] + 1
    syn_occ = np.full((T1, n_syn), -1, dtype=np.int32)
    for r in syn_dw.itertuples(index=False):
        syn_occ[r.bind_step:r.unbind_step, r.synthetic_index] = r.molecule_id

    # ---- labelled molecules: every physics step ---------------------------
    st, bs = state_timeline(res, labeled, n_steps)
    P = labeled_positions(res, exposure_data, labeled, n_steps, spd, spe)
    x_px, y_px, zk, dz = to_pixels(P, geom)
    steps = np.arange(n_steps)
    if focus_z is not None:
        fz_step = focus_z[np.minimum(steps // spd, len(focus_z) - 1)]
        dz = P[:, :, 2] - fz_step[:, None]
        zk = np.zeros(dz.shape, dtype=int)
    synth_nm = res['synth_pos'][:n_steps].astype(float) * 1000.0
    if n_syn:
        d = np.linalg.norm(P[:, :, None, :] - synth_nm[:, None, :, :], axis=-1)
        dist_syn = np.nanmin(np.where(np.isnan(d), np.inf, d), axis=-1)
        dist_syn[np.isnan(P[:, :, 0])] = np.nan
    else:
        dist_syn = np.full(st.shape, np.nan)
    is_syn = res['is_synthetic']
    on_syn = np.where(bs >= 0, is_syn[np.maximum(bs, 0)], False)
    L = len(labeled)
    track = pd.DataFrame({
        'step': np.repeat(steps, L),
        'time_s': np.repeat(steps * dt, L),
        'frame': np.repeat(steps // spd, L),
        'in_exposure': np.repeat((steps % spd) < spe, L),
        'molecule_id': np.tile(labeled, n_steps),
        'state': st.ravel(),
        'state_label': pd.Series(st.ravel()).map(STATE_LABELS).values,
        'bound_site': bs.ravel(),
        'bound_to_synthetic': on_syn.ravel(),
        'x_nm': P[:, :, 0].ravel(), 'y_nm': P[:, :, 1].ravel(),
        'z_nm': P[:, :, 2].ravel(),
        'x_px': x_px.ravel(), 'y_px': y_px.ravel(),
        'nearest_slice': np.where(np.isnan(P[:, :, 2]), -1, zk).ravel(),
        'dz_to_slice_nm': dz.ravel(),
        'dist_to_synthetic_nm': dist_syn.ravel(),
    })
    if focus_z is not None:
        track['focal_z_nm'] = np.repeat(fz_step, L)
    track.to_csv(os.path.join(tdir, 'labeled_track.csv'), index=False,
                 float_format='%.3f')

    # ---- labelled molecules: per camera frame -----------------------------
    ex = track[track['in_exposure']]
    g = ex.groupby(['frame', 'molecule_id'], sort=True)
    frames_df = g.agg(x_nm=('x_nm', 'mean'), y_nm=('y_nm', 'mean'),
                      z_nm=('z_nm', 'mean'),
                      frac_bound=('state', lambda s: float(np.mean(s == 2))),
                      frac_on_synthetic=('bound_to_synthetic', 'mean'),
                      min_dist_to_synthetic_nm=('dist_to_synthetic_nm', 'min'),
                      modal_state=('state', lambda s: int(pd.Series(s).mode().iloc[0])),
                      ).reset_index()
    frames_df.insert(1, 'time_s', frames_df['frame'] * spd * dt)
    fx, fy, fk, fdz = to_pixels(frames_df[['x_nm', 'y_nm', 'z_nm']].values, geom)
    if focus_z is not None:
        _fz = focus_z[np.minimum(frames_df['frame'].values, len(focus_z) - 1)]
        fdz = frames_df['z_nm'].values - _fz
        fk = np.zeros(len(frames_df), dtype=int)
        frames_df['focal_z_nm'] = _fz
    frames_df['x_px'], frames_df['y_px'] = fx, fy
    frames_df['nearest_slice'], frames_df['dz_to_slice_nm'] = fk, fdz
    frames_df['modal_state_label'] = frames_df['modal_state'].map(STATE_LABELS)
    arr_c = array_centroid_per_frame(res, num_frames, spd, spe)
    if arr_c is not None:
        _ac = arr_c[np.minimum(frames_df['frame'].values, len(arr_c) - 1)]
        ax_px, ay_px, _, _ = to_pixels(_ac, geom)
        frames_df['array_x_px'], frames_df['array_y_px'] = ax_px, ay_px
        frames_df['array_z_nm'] = _ac[:, 2]
    frames_df.to_csv(os.path.join(tdir, 'labeled_frames.csv'), index=False,
                     float_format='%.3f')

    # ---- neighbourhood of the synthetic array, per frame (all molecules) --
    if n_syn and exposure_data is not None:
        rows = []
        for f in range(int(num_frames)):
            exp = exposure_data.get(f)
            if exp is None:
                continue
            Pall = (np.asarray(exp['pos_steps']) if isinstance(exp, dict)
                    else np.array([s['pos'] for s in exp])) * 1000.0
            k0 = f * spd
            n_near, lab_near = [], []
            for j in range(len(Pall)):
                k = min(k0 + j, len(res['synth_pos']) - 1)
                tree = cKDTree(res['synth_pos'][k].astype(float) * 1000.0)
                dd, _ = tree.query(Pall[j], k=1)
                near = dd <= near_radius_nm
                n_near.append(int(near.sum()))
                lab_near.append(int(near[labeled].sum()))
            occ = syn_occ[k0:k0 + len(Pall)]
            occupants = sorted(set(occ[occ >= 0].tolist()))
            rows.append({
                'frame': f, 'time_s': k0 * dt,
                'mean_molecules_within_r': float(np.mean(n_near)),
                'max_molecules_within_r': int(np.max(n_near)),
                'labeled_within_r_any': bool(np.any(lab_near)),
                'mean_synthetic_sites_occupied': float(np.mean((occ >= 0).sum(axis=1))),
                'occupant_ids': ';'.join(str(o) for o in occupants),
                'labeled_occupant': bool(np.isin(occupants, labeled).any()) if occupants else False,
            })
        pd.DataFrame(rows).to_csv(os.path.join(tdir, 'synthetic_neighborhood.csv'),
                                  index=False, float_format='%.3f')

    # ---- raw arrays ------------------------------------------------------
    npz = {'synthetic_site_ids': res['synthetic_site_ids'],
           'synthetic_pos_nm': res['synth_pos'].astype(np.float32) * 1000.0,
           'synthetic_occupant': syn_occ,
           'labeled_ids': labeled,
           'is_synthetic': res['is_synthetic'],
           'site_strengths': res['site_strengths']}
    if focus_z is not None:
        npz['smt_focus_z_nm'] = focus_z
        pd.DataFrame({'frame': np.arange(len(focus_z)),
                      'time_s': np.arange(len(focus_z)) * spd * dt,
                      'focal_z_nm': focus_z}).to_csv(
            os.path.join(tdir, 'smt_focus_track.csv'), index=False,
            float_format='%.3f')
    if 'traj_step' in res:
        npz.update(traj_step=res['traj_step'],
                   traj_pos_nm=res['traj_pos'] * np.float32(1000.0),
                   traj_states=res['traj_states'], traj_bound=res['traj_bound'])
    np.savez(os.path.join(tdir, 'tracking_arrays.npz'), **npz)

    # ---- SMT images ------------------------------------------------------
    smt_display_range = None
    cross_on = False
    if write_smt:
        sp = dict(smt_params or {})
        bit_depth = 8 if int(bit_depth) == 8 else 16
        sdir = os.path.join(row_outdir, 'SMT_%dbit' % bit_depth)
        os.makedirs(sdir, exist_ok=True)
        from tifffile import imwrite
        stacks, frame_geoms = {}, {}
        for f in range(int(num_frames)):
            W = _exposure_window_positions(exposure_data, f, labeled)
            if W is None:
                continue
            frame_geoms[f] = (dict(geom, z_slices_nm=[float(focus_z[f])])
                              if focus_z is not None else geom)
            stacks[f] = render_smt_frame(
                W * 1000.0, frame_geoms[f], lateral_fwhm_nm, axial_fwhm_nm,
                photons=sp.get('photons', 300.0),
                background=sp.get('background', 10.0),
                gain=sp.get('gain', 1.0), offset=sp.get('offset', 100.0),
                read_noise=sp.get('read_noise', 1.5),
                add_noise=sp.get('add_noise', True), rng=label_rng)
        if stacks:
            # One display range for the whole movie, computed from the
            # rendered data BEFORE any marker is drawn.
            allpx = np.concatenate([st_.ravel() for st_ in stacks.values()])
            lo = float(np.percentile(allpx, 0.5))
            hi = float(allpx.max())
            if hi <= lo:
                hi = lo + 1.0
            smt_display_range = (lo, hi)
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
                    c = arr_c[min(f, len(arr_c) - 1)]
                    cx, cy, _, _ = to_pixels(c, geom)
                    zs = np.asarray(frame_geoms[f]['z_slices_nm'], dtype=float)
                    if len(zs):
                        near = np.abs(zs - c[2]) <= axial_fwhm_nm
                        if not near.any():
                            near[np.argmin(np.abs(zs - c[2]))] = True
                        for zi in np.where(near)[0]:
                            draw_open_cross(out[zi], cx, cy, mark)
                imwrite(os.path.join(sdir, 'frame_%04d.tiff' % f), out,
                        imagej=True, metadata=meta, compression='zlib')
            log("  Tracking: SMT images -> SMT_%dbit/ (%s%s)" % (
                bit_depth,
                'raw camera ADU %d-%d, display range embedded' % (int(allpx.min()), int(hi))
                if bit_depth == 16 else
                'contrast-stretched from %d-%d ADU; use --bit-depth 16 for raw counts' % (int(lo), int(hi)),
                '; array centroid marked with a cross' if cross_on else ''))

    summary = {
        'label_spec': str(label_spec),
        'labeled_ids': [int(i) for i in labeled],
        'label_note': note,
        'num_molecules': int(res['num_molecules']),
        'num_sites': int(res['num_sites']),
        'synthetic_site_ids': [int(i) for i in res['synthetic_site_ids']],
        'sim_dt_s': dt, 'steps_per_display': int(spd),
        'steps_per_exposure': int(spe), 'num_frames': int(num_frames),
        'n_binding_intervals': int(len(dwells)),
        'n_synthetic_binding_intervals': int(len(syn_dw)),
        'n_labeled_synthetic_binding_intervals': int(syn_dw['labeled'].sum()),
        'n_synthetic_visits_merged': int(len(visits)),
        'n_labeled_synthetic_visits_merged': int(visits['labeled'].sum()),
        'synthetic_binder_ids': sorted(int(i) for i in syn_dw['molecule_id'].unique()),
        'image_geometry': geom,
        'smt_rendered': bool(write_smt),
        'smt_params': smt_params if write_smt else None,
        'smt_focus_mode': focus_mode if (focus_mode != 'track' or focus_z is not None) else 'slices',
        'smt_focus_z_range_nm': ([float(focus_z.min()), float(focus_z.max())]
                                 if focus_z is not None else None),
        'smt_display_range_adu': smt_display_range,
        'smt_bit_depth': (8 if int(bit_depth) == 8 else 16) if write_smt else None,
        'smt_cross_drawn': cross_on,
        'near_radius_nm': near_radius_nm,
        'pixel_convention': ('pixel (c, r) covers [c, c+1) x [r, r+1); '
                             'centre at +0.5; row 0 = top (largest y)'),
    }
    with open(os.path.join(tdir, 'tracking_summary.json'), 'w') as fh:
        json.dump(summary, fh, indent=2)
    log("  Tracking: labelled %s | %d binding intervals (%d on synthetic "
        "array, %d by labelled) -> %s" % (
            summary['labeled_ids'], summary['n_binding_intervals'],
            summary['n_synthetic_binding_intervals'],
            summary['n_labeled_synthetic_binding_intervals'], tdir))
    return labeled