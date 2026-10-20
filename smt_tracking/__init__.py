"""
smt_tracking - Binding-event recording and single-molecule-tracking (SMT)
image simulation on top of SPARK.

1. BindingTracker is handed to spark's run_full_simulation(tracker=...) and
   logs every state / binding change of every molecule.
2. One or more molecules are "labelled" (pure bookkeeping: all molecules are
   simulated identically).
3. write_tracking_outputs() writes binding tables, the labelled molecules'
   tracks and SMT camera images showing only the labelled molecules.

Modules
-------
labels      label specs ("random:N", "ids:a,b", "synthetic:N"), label RNG
tracker     BindingTracker (observer called at every physics step)
events      event / dwell / visit tables rebuilt from the event log
geometry    pixel coordinates (same grid as spark's images), focal planes
positions   labelled-molecule positions, synthetic-array centroid / depth
rendering   SMT camera image of one frame, array marker
images      writing the SMT image stacks
outputs     write_tracking_outputs(): the Tracking/ folder
"""

from .labels import parse_label_spec, make_label_rng, resolve_labels
from .tracker import BindingTracker
from .geometry import image_geometry, to_pixels
from .outputs import write_tracking_outputs
