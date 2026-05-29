# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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

# Force-disable Ray dashboard on low-podPidsLimit clusters. nv-ingest's
# entrypoint calls ray.init(dashboard_host=..., dashboard_port=...) explicitly,
# which overrides the RAY_DISABLE_DASHBOARD env var. The dashboard spawns ~10
# subprocesses carrying ~50 threads each, collectively ~500 PIDs. Patching
# ray.init here forces include_dashboard=False before any worker is spawned.
try:
    import ray as _ray
    _orig_ray_init = _ray.init

    def _patched_ray_init(*args, **kwargs):
        kwargs["include_dashboard"] = False
        return _orig_ray_init(*args, **kwargs)

    _ray.init = _patched_ray_init
except Exception:
    pass
