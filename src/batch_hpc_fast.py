#!/usr/bin/env python3
"""
batch_hpc_fast.py - Command-line entry point of minimal SPARK.

Every CSV row is one simulation. Every row and every seed is simulated from
scratch and is statistically independent: no simulation is cached or reused
between rows or between runs. The work for one row is in spark/pipeline.py (project root).

  Single node:   python batch_hpc_fast.py params.csv -o results/ -j 16
  SLURM array:   one task per row, e.g. --rows $SLURM_ARRAY_TASK_ID

RANDOM STREAMS
  --seed S gives row i the stream SeedSequence([S, i]); without --seed each
  run uses fresh entropy, recorded in run_info.json. Sites and molecules
  always use separate streams. --shared-sites gives every row of a seed the
  same site stream (paired comparisons, e.g. WT vs DBD on the same locus).

CONVOLVED IMAGES
  Default: absolute units (AU calibration, au_* CSV columns); every row is
  rendered on its own.
  --auto-normalize: every row is divided by the same vmax, the brightest
  voxel over ALL rows of the run times --headroom (display_noise_* CSV
  columns add optional noise). This needs every row simulated before any
  is rendered, so it runs in two phases:
    phase 1  simulate the row, write everything except the convolved images,
             keep its 3D grids (_grids.npz) and its maximum (_max.txt)
    phase 2  vmax from all rows' _max.txt, render Convolved_auto_<bit>bit/
  Without --phase both run in one call. On SLURM submit one array with
  --phase 1, then one with --phase 2 after it (afterok dependency); phase 2
  refuses to start until every CSV row has its _max.txt.
"""

import argparse
import os
import sys
import time
from multiprocessing import Pool, cpu_count

import pandas as pd

# The packages spark/ and smt_tracking/ live in the project root, one level
# above this script.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from spark.params import DEFAULTS
from spark.pipeline import process_row, render_normalized_row, phase2_vmax
from spark.outputs import write_calibration


# ============================================================================
# COMMAND LINE
# ============================================================================

def build_parser():
    parser = argparse.ArgumentParser(
        description='Minimal SPARK batch simulation (independent rows/seeds)',
        epilog=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('csv_file', help='Input CSV (one row per simulation)')

    g = parser.add_argument_group('run')
    g.add_argument('-o', '--outdir', default='./batch_output')
    g.add_argument('-j', '--jobs', type=int, default=None,
                   help='Parallel workers (default: all CPUs)')
    g.add_argument('--rows', default=None, help='Row range: "5" or "0-9" (default: all)')
    g.add_argument('--seed', type=int, default=None,
                   help='Base seed; row i uses SeedSequence([seed, i]). '
                        'Omit for fresh, non-reproducible randomness.')
    g.add_argument('--shared-sites', action='store_true',
                   help='All rows of this seed share one site stream (same '
                        'layout and site motion for equal site parameters); '
                        'molecules stay independent per row. Requires --seed.')

    g = parser.add_argument_group('images and tables')
    g.add_argument('--bit-depth', type=int, choices=[8, 16], default=8)
    g.add_argument('--no-conv', action='store_true',
                   help='Do not write the all-molecule convolved images')
    g.add_argument('--auto-normalize', action='store_true',
                   help='Scale convolved images of all rows by one shared '
                        'maximum instead of absolute units')
    g.add_argument('--headroom', type=float, default=1.0,
                   help='Auto-normalize vmax = global max * headroom')
    g.add_argument('--phase', type=int, choices=[1, 2], default=None,
                   help='With --auto-normalize on SLURM: 1 = simulate one row '
                        'and keep its grids, 2 = render it with the global maximum')
    g.add_argument('--upscale-factor', type=int, default=1,
                   help='Pixel repetition for convolved images (default 1)')
    g.add_argument('--save-gt', action='store_true',
                   help='Save ground-truth renderings')
    g.add_argument('--save-positions', action='store_true',
                   help='Save per-molecule ground-truth positions per frame as a '
                        'tab-delimited ViSP-compatible file '
                        '(x[nm] y[nm] z[nm] intensity frame state)')
    g.add_argument('--positions-filename', default='positions.csv')

    g = parser.add_argument_group('binding tracking / single-molecule tracking')
    g.add_argument('--record-binding', action='store_true',
                   help='Record every binding / state change of every molecule '
                        '(Tracking/ folder: state_events, site_dwells, '
                        'synthetic_dwells, synthetic_visits, labelled tracks).')
    g.add_argument('--smt', action='store_true',
                   help='Also render SMT images (SMT_<bit>bit/) showing ONLY the '
                        'labelled molecule(s). Implies --record-binding.')
    g.add_argument('--smt-label', default='random:1',
                   help='Which molecules are labelled: "random:N", "ids:3,17" '
                        'or "synthetic:N" (N molecules that bound the '
                        'synthetic array). A per-row CSV column "smt_label" '
                        'overrides this. Default: random:1')
    g.add_argument('--smt-photons', type=float, default=300.0,
                   help='Photons per labelled molecule per exposure, in focus '
                        '(default 300)')
    g.add_argument('--smt-background', type=float, default=10.0,
                   help='Background photons per pixel per exposure (default 10)')
    g.add_argument('--smt-gain', type=float, default=1.0,
                   help='Camera gain, ADU per photoelectron (default 1)')
    g.add_argument('--smt-offset', type=float, default=100.0,
                   help='Camera baseline in ADU (default 100)')
    g.add_argument('--smt-read-noise', type=float, default=1.5,
                   help='Read noise, e- rms (default 1.5)')
    g.add_argument('--smt-no-cross', action='store_true',
                   help='Do not draw the cross marking the synthetic array '
                        'centroid (use for clean images to analyse; the cross '
                        'overwrites pixel values)')
    g.add_argument('--smt-no-noise', action='store_true',
                   help='Noise-free SMT images (expected values only)')
    g.add_argument('--smt-focus', choices=['slices', 'synthetic', 'track'],
                   default='slices',
                   help='"slices" (default): SMT z-planes follow the row\'s '
                        'slice settings. "synthetic": one fixed focal plane '
                        'at the depth of the synthetic array at t = 0. '
                        '"track": one focal plane that follows the array, '
                        're-centred on its mean depth every frame (focus lock).')
    g.add_argument('--near-radius-nm', type=float, default=250.0,
                   help='Radius around synthetic sites for '
                        'synthetic_neighborhood.csv (default 250 nm)')
    g.add_argument('--trajectory-stride', type=int, default=0,
                   help='Also save positions/states/bound site of ALL '
                        'molecules every N physics steps (0 = off; large!)')
    return parser


def check_args(parser, args):
    if args.shared_sites and args.seed is None:
        parser.error('--shared-sites requires --seed')
    if args.phase is not None and not args.auto_normalize:
        parser.error('--phase is only needed with --auto-normalize; without it '
                     'just run each row normally')
    if args.auto_normalize and args.no_conv:
        parser.error('--auto-normalize applies to convolved images; drop --no-conv')


def build_config(args):
    """Settings shared by all rows (passed to every worker)."""
    return {
        'outdir': args.outdir,
        'seed': args.seed,
        'shared_sites': args.shared_sites,
        'bit_depth': args.bit_depth,
        'save_conv': not args.no_conv,
        'auto_normalize': args.auto_normalize,
        'upscale_factor': args.upscale_factor,
        'save_gt': args.save_gt,
        'save_positions': args.save_positions,
        'positions_filename': args.positions_filename,
        'record_binding': args.record_binding or args.smt,
        'smt': args.smt,
        'smt_label': args.smt_label,
        'smt_params': {'photons': args.smt_photons,
                       'background': args.smt_background,
                       'gain': args.smt_gain, 'offset': args.smt_offset,
                       'read_noise': args.smt_read_noise,
                       'add_noise': not args.smt_no_noise,
                       'focus': args.smt_focus,
                       'cross': not args.smt_no_cross},
        'near_radius_nm': args.near_radius_nm,
        'trajectory_stride': args.trajectory_stride,
    }


def load_csv(path):
    """CSV rows, with DEFAULTS filled in for missing columns."""
    df = pd.read_csv(path)
    for key, val in DEFAULTS.items():
        if key not in df.columns:
            df[key] = val
    return df


def parse_rows(rows_str, total_rows):
    """Parse '5' -> [5], '0-9' -> [0..9], None -> all."""
    if rows_str is None:
        return list(range(total_rows))
    if '-' in rows_str:
        start, end = rows_str.split('-', 1)
        return list(range(int(start), min(int(end) + 1, total_rows)))
    return [int(rows_str)]


def run_parallel(func, items, n_workers):
    if n_workers > 1:
        with Pool(n_workers) as pool:
            return pool.map(func, items)
    return [func(item) for item in items]


def print_header(args, config, batch_df, row_indices, n_workers):
    conv = ("off" if not config['save_conv'] else
            "auto-normalized (headroom=%.2f)" % args.headroom
            if args.auto_normalize else "AU")
    print("=" * 60)
    print("SPARK minimal batch")
    print("  CSV:        %s (%d total rows)" % (args.csv_file, len(batch_df)))
    print("  Rows:       %d-%d (%d rows)" % (row_indices[0], row_indices[-1], len(row_indices)))
    print("  Workers:    %d" % n_workers)
    print("  Output:     %s" % args.outdir)
    print("  Seed:       %s%s" % (args.seed if args.seed is not None else "fresh entropy",
                                  "  (sites shared across rows)" if args.shared_sites else ""))
    print("  Convolved:  %s, %d-bit" % (conv, args.bit_depth))
    print("  GT:         %s" % args.save_gt)
    print("  Positions:  %s" % args.save_positions)
    print("  Tracking:   %s%s" % (config['record_binding'],
          ("  (SMT images, default label=%s; CSV smt_label overrides)" % args.smt_label)
          if args.smt else ""))
    print("=" * 60)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = build_parser()
    args = parser.parse_args()
    check_args(parser, args)

    batch_df = load_csv(args.csv_file)
    row_indices = parse_rows(args.rows, len(batch_df))
    os.makedirs(args.outdir, exist_ok=True)
    config = build_config(args)
    items = [(idx, batch_df.iloc[idx].to_dict(), config) for idx in row_indices]

    # ---- SLURM array mode (auto-normalize): this task's rows only --------
    if args.phase == 1:
        for item in items:
            process_row(item)
        return
    if args.phase == 2:
        try:
            config['auto_vmax'] = phase2_vmax(batch_df, args.outdir,
                                              args.headroom, args.bit_depth)
        except RuntimeError as e:
            print("ERROR: %s" % e, file=sys.stderr)
            sys.exit(1)
        for item in items:
            render_normalized_row(item)
        return

    # ---- Single node: all requested rows in this call -------------------
    n_workers = args.jobs if args.jobs else min(cpu_count(), len(row_indices))
    print_header(args, config, batch_df, row_indices, n_workers)
    t_start = time.time()

    results = run_parallel(process_row, items, n_workers)

    if args.auto_normalize:
        # Same 10-decimal rounding as the _max.txt files of --phase 1/2, so
        # single-node and SLURM runs give identical images.
        global_max = max(float('%.10f' % r[2]) for r in results)
        config['auto_vmax'] = write_calibration(
            args.outdir, global_max, args.headroom, args.bit_depth)
        print("\nGlobal max: %.6f  Auto vmax: %.6f" % (global_max, config['auto_vmax']))
        run_parallel(render_normalized_row, items, n_workers)

    print("\n" + "=" * 60)
    print("All done! %d rows in %.1fs" % (len(row_indices), time.time() - t_start))
    print("Output: %s" % os.path.abspath(args.outdir))
    print("=" * 60)


if __name__ == '__main__':
    main()
