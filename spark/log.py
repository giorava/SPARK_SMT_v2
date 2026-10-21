"""
log.py - Timestamped log lines with the process's peak memory.
"""

import resource
import sys
import time


def _mem_gb():
    """Peak resident memory of this process in GB (ru_maxrss is in bytes on
    macOS, in KB on Linux)."""
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 ** 3 if sys.platform == 'darwin' else 1024 ** 2)


def log(msg):
    """Print with timestamp and memory."""
    print("[%s | %.1f GB] %s" % (time.strftime("%H:%M:%S"), _mem_gb(), msg), flush=True)
