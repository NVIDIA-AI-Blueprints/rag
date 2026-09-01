# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

"""
aiperf endpoint plugin for the NVIDIA RAG Blueprint server.

Registration
------------
aiperf discovers plugins via setuptools entry points.  After installing
rag-perf (``pip install -e .``), the ``nvidia_rag`` endpoint type is
automatically available to aiperf::

    aiperf profile --endpoint-type nvidia_rag ...

The entry-point declaration is in ``pyproject.toml``:

.. code-block:: toml

    [project.entry-points."aiperf.plugins"]
    nvidia_rag = "rag_perf.plugin:plugins.yaml"

Fallback registration
---------------------
If the entry-point mechanism is not yet supported by the installed aiperf
version, call ``register()`` before invoking any aiperf API::

    from rag_perf.plugin import register
    register()

This patches aiperf's in-memory plugin registry directly.
"""

from rag_perf.plugin.nvidia_rag import (
    PLUGIN_CLASS_PATH,
    PLUGIN_NAME,
    NvidiaRagEndpoint,
    PluginRegistry,
)

register = PluginRegistry.register

__all__ = [
    "NvidiaRagEndpoint",
    "PLUGIN_CLASS_PATH",
    "PLUGIN_NAME",
    "PluginRegistry",
    "register",
]
