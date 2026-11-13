"""Memory safeguards: a whole day's flights over a continent can fill a 16 GB Mac, and a full Mac freezes (the owner's
whole-US date-only job went from 36 % to 15 % free in two minutes on 8 workers). The heavy stages watch the free memory
and give back workers before that point; the app stops a job outright as a last resort."""
import subprocess
import sys
import threading
from contextlib import contextmanager

SOFT_PCT = 25  # below this, a pool's workers are stopped and the stage goes on with half as many
HARD_PCT = 12  # below this, the app stops the job


def free_pct():
    """The share of memory still available, % (macOS kern.memorystatus_level, Linux MemAvailable); None where it
    can't be read."""
    try:
        if sys.platform.startswith("linux"):
            m = dict(line.split(":", 1) for line in open("/proc/meminfo"))
            return int(100 * int(m["MemAvailable"].split()[0]) / int(m["MemTotal"].split()[0]))
        return int(subprocess.run(["sysctl", "-n", "kern.memorystatus_level"], capture_output=True, text=True, timeout=2).stdout)
    except (OSError, ValueError, KeyError, subprocess.SubprocessError):
        return None


@contextmanager
def guard(ex, floor=SOFT_PCT, every_s=1.0):
    """While the block runs, kill the worker processes of the ProcessPoolExecutor ex as soon as free memory falls below
    floor %: the pool's map then raises BrokenProcessPool. Yields a list that holds the free % seen when it fired."""
    stop, hit = threading.Event(), []

    def watch():
        while not stop.wait(every_s):
            f = free_pct()
            if f is not None and f < floor:
                hit.append(f)
                for p in list((ex._processes or {}).values()):  # shortcut: a private attribute (CPython 3.8-3.13)
                    p.kill()
                return

    th = threading.Thread(target=watch, daemon=True)
    th.start()
    try:
        yield hit
    finally:
        stop.set()
        th.join()
