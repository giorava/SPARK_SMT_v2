"""
events.py - Tables rebuilt from the tracker's event log.

STATE CODES:  0 free, 1 intermediate, 2 specifically bound, 3 non-specific.
SITE IDS:     index into the engine's site array. Synthetic sites are the
              LAST num_synthetic_sites entries; synthetic_index 0..n-1 gives
              their order within the array.
"""

import numpy as np
import pandas as pd


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
