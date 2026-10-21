"""
tracker.py - BindingTracker, the observer handed to run_full_simulation().

At every physics step it compares each molecule's (state, bound site)
with the previous step and logs ONLY the changes. From that
event log the exact state and bound site of every molecule at every step
can be rebuilt (see events.py), so nothing is lost but storage stays small.
It only reads the simulation state and never draws from the simulation's
random streams.

STEP / TIME CONVENTION
Step k is the snapshot taken BEFORE physics step k runs, at time k * sim_dt.
This is the same snapshot the engine stores in its exposure buffers, so:
    frame       = k // steps_per_display
    in_exposure = (k % steps_per_display) < steps_per_exposure
An event logged at step k first becomes visible in snapshot k (it happened
during physics step k-1). A molecule bound from bind_step to unbind_step is
seen bound in snapshots bind_step .. unbind_step-1, so dwell = (unbind_step -
bind_step) * sim_dt.
"""

import numpy as np

from .labels import parse_label_spec, make_label_rng


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

    def observe(self, step, pos, states, sites, bound_site_indices):
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
            ps, pb = self._prev
            # A molecule changes at most once per physics step, so a change
            # of state or bound site captures every event.
            changed = (states != ps) | (bound != pb)
            idx = np.nonzero(changed)[0]
            if len(idx):
                self._append(np.full(len(idx), step), idx, ps[idx],
                             states[idx], pb[idx], bound[idx])
        self._prev = (states.copy(), bound.copy())

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
