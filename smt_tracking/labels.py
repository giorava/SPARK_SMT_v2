"""
labels.py - Which molecules are fluorescently labelled.

Labelling is pure bookkeeping: all molecules are simulated identically and
only the labelled ones are drawn in the SMT images, so the label can even
be chosen after the simulation (e.g. "a molecule that visited the
synthetic array").

Label specs (case-insensitive):
    "random:N"     N molecules chosen at random
    "ids:3,17"     explicit molecule ids
    "synthetic:N"  N molecules among those that bound the synthetic array
"""

import numpy as np

from .events import dwell_table


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
