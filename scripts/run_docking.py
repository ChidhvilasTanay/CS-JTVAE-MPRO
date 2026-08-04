"""Keep-awake wrapper: dock the 6-obj NSGA-III res.opt Pareto front via
run_docking.main. Prevents laptop sleep during the long Vina run. Writes to
data/analogs_v3_6obj_docked.csv (a COPY); the protected resopt file is untouched.
"""
import ctypes
from run_docking import main

ES_CONTINUOUS = 0x80000000
ES_SYSTEM_REQUIRED = 0x00000001


def _keep_awake():
    try:
        ctypes.windll.kernel32.SetThreadExecutionState(
            ES_CONTINUOUS | ES_SYSTEM_REQUIRED
        )
    except Exception:
        pass


if __name__ == "__main__":
    _keep_awake()
    raise SystemExit(main("configs/docking_v3_6obj.yaml"))
