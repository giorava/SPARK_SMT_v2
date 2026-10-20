"""
seeds.py - Random streams of one CSV row.

Every row gets its own numpy SeedSequence, split into independent child
streams for site landscape/motion, molecules, image noise and labelling.
"""

from collections import namedtuple

import numpy as np


_SITE_STREAM = 0x5173   # tag separating the shared site stream from rows

RowRNGs = namedtuple('RowRNGs', [
    'seed_sequence',  # the row's SeedSequence (its entropy reproduces the row)
    'site',           # Generator: site layout, strengths, site motion
    'mol',            # Generator: molecules
    'render',         # Generator: noise in the all-molecule images
    'label_seed',     # int: labelling / SMT camera noise (smt_tracking)
])


def row_seed_sequence(base_seed, row_idx):
    """Independent SeedSequence for one row.

    With a base seed the stream is SeedSequence([base_seed, row_idx]); this
    never overlaps another (seed, row) pair, unlike base_seed + row_idx
    (where seed 1 / row 1 would equal seed 2 / row 0). Without a base seed
    the stream comes from fresh OS entropy.
    """
    if base_seed is None:
        return np.random.SeedSequence()
    return np.random.SeedSequence([int(base_seed), int(row_idx)])


def shared_site_seed_sequence(base_seed):
    """Site stream shared by ALL rows of one seed (--shared-sites).

    Every row then gets the same site layout, strengths and site motion
    (for equal site parameters), while molecules stay independent per row.
    """
    return np.random.SeedSequence([int(base_seed), _SITE_STREAM])


def _children(ss):
    """Child streams of a row: (site, molecule, image noise, labelling)."""
    return ss.spawn(4)


def row_rngs(base_seed, row_idx, shared_sites=False):
    """All random streams of one row."""
    ss = row_seed_sequence(base_seed, row_idx)
    ss_site, ss_mol, ss_render, ss_label = _children(ss)
    if shared_sites:
        ss_site = shared_site_seed_sequence(base_seed)
    return RowRNGs(ss,
                   np.random.default_rng(ss_site),
                   np.random.default_rng(ss_mol),
                   np.random.default_rng(ss_render),
                   int(ss_label.generate_state(1)[0]))


def render_rng_from_entropy(entropy):
    """Rebuild a row's image-noise Generator from its saved entropy
    (run_info.json), e.g. for auto-normalize phase 2."""
    return np.random.default_rng(_children(np.random.SeedSequence(entropy))[2])
