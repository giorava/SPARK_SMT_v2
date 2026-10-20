"""
SPARK (minimal) - particle simulation of transcription-factor binding in a
spherical nucleus, with virtual-microscope imaging.

Modules
-------
params        defaults and parsing of one CSV row
sites         binding-site layout and strengths
site_motion   fractional-Brownian motion of site clusters
physics       one physics step (transitions, site motion, (un)binding, diffusion)
simulation    run_full_simulation(): set-up and time loop
exposure      per-frame summaries of the camera exposure window
geometry      image grid, z-planes, pixel coordinates
convolution   PSF-convolved all-molecule images (AU or normalized)
ground_truth  ground-truth renderings
smt_tracking  binding-event recording and single-molecule (SMT) images

seeds         random streams of one row (sites / molecules / noise / labels)
pipeline      everything done for one CSV row (used by batch_hpc_fast.py)
outputs       files written per row (TIFFs, hyperstacks, tables, run info)
log           timestamped log lines
"""
