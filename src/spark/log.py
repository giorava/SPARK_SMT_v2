"""
log.py - Timestamped log lines with the process's peak memory.
"""

import resource
import time


def _mem_gb():
    """Peak resident memory of this process in GB (macOS reports bytes,
    Linux KB; the value is indicative only)."""
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 ** 2)


def log(msg):
    """Print with timestamp and memory."""
    print("[%s | %.1f GB] %s" % (time.strftime("%H:%M:%S"), _mem_gb(), msg), flush=True)
