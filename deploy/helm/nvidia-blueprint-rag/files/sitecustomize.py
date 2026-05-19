import os

_n = max(1, int(os.environ.get("RAG_NV_INGEST_DETECTED_CPUS", "1")))
_cpu_set = set(range(_n))

os.cpu_count = lambda: _n
try:
    os.sched_getaffinity = lambda _pid=0: set(_cpu_set)
except Exception:
    pass

try:
    import psutil
    psutil.cpu_count = lambda *a, **k: _n
except Exception:
    pass
