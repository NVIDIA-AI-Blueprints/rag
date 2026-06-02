# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

import os
import shutil
from pathlib import Path

_DOCS_DIR = Path(__file__).resolve().parent
_VERSIONS_JSON = _DOCS_DIR / "versions1.json"


def _copy_versions_json(outdir: Path) -> Path | None:
    """Copy the shared switcher manifest beside the version output folder.

    Published and local preview layouts look like::

        _build/
          versions1.json      <- one shared file (not copied into each release)
          2.6.0/              <- HTML for the current release (sphinx outdir)
          2.5.0/              <- prior release builds (from earlier publishes)
          2.4.0/
          2.3.0/

    The theme loads ``../versions1.json`` from pages under ``2.6.0/``, so the
    JSON must live in ``_build/``, not inside each release folder.
    """
    if not _VERSIONS_JSON.is_file():
        return None
    dest = outdir.parent / "versions1.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    # copyfile (not copy2) so the destination mtime reflects the build time.
    # copy2 preserved the source timestamp and made the output look stale.
    shutil.copyfile(_VERSIONS_JSON, dest)
    return dest


def setup(app):
    def _on_builder_inited(_app) -> None:
        _copy_versions_json(Path(_app.outdir))

    def _on_build_finished(_app, exc) -> None:
        if exc is not None:
            return
        dest = _copy_versions_json(Path(_app.outdir))
        if dest is not None:
            print(
                f"\nVersion switcher manifest: {dest}\n"
                f"  Upload this file to the docs site root (e.g. rag/versions1.json).\n"
                f"  It is intentionally NOT inside {_app.outdir}.\n"
            )

    app.connect("builder-inited", _on_builder_inited)
    app.connect("build-finished", _on_build_finished)
    return {"version": "1.0", "parallel_read_safe": True}


project = " NVIDIA RAG blueprint"
copyright = "2025, NVIDIA CORPORATION & AFFILIATES"
author = "NVIDIA CORPORATION & AFFILIATES"
release = "2.6.0"

# Sphinx outdir should be ``_build/<release>/`` (for example ``_build/2.6.0/``).
# See documentation.md for the version-switcher layout.
DOCS_HTML_OUTDIR = f"_build/{release}"

# Shared parent-level manifest for the PyData/NVIDIA version switcher.
# Override with DOCS_SWITCHER_JSON=versions1.json when using sphinx-autobuild,
# which serves only the outdir and cannot reach ../versions1.json.
_switcher_json_url = os.environ.get("DOCS_SWITCHER_JSON", "../versions1.json")

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "myst_parser",  # For our markdown docs
    "sphinx.ext.viewcode",  # For adding a link to view source code in docs
    "sphinx.ext.doctest",  # Allows testing in docstrings
    "sphinx.ext.napoleon",  # For google style docstrings
    "sphinx_copybutton",  # For copy button in code blocks
    "swagger_plugin_for_sphinx", # For parsing and presenting OpenAPI specs
    "sphinx_reredirects",  # For handling URL redirects
]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

# -- Options for MyST Parser (Markdown) --------------------------------------
# MyST Parser settings
myst_enable_extensions = [
    "dollarmath",  # Enables dollar math for inline math
    "amsmath",  # Enables LaTeX math for display mode
    "colon_fence",  # Enables code blocks using ::: delimiters instead of ```
    "deflist",  # Supports definition lists with term: definition format
    "fieldlist",  # Enables field lists for metadata like :author: Name
    "tasklist",  # Adds support for GitHub-style task lists with [ ] and [x]
]
myst_heading_anchors = 5  # Generates anchor links for headings up to level 5

# -- Options for Redirects ---------------------------------------------------
# Configure redirects for renamed files
redirects = {
    "nemoretriever-parse-extraction": "nemotron-parse-extraction.html",
}

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "nvidia_sphinx_theme"
html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA-AI-Blueprints/rag",
            "icon": "fa-brands fa-github",
        }
    ],
    "switcher": {"json_url": _switcher_json_url, "version_match": release},
    "extra_head": {
        """
    <script src="https://assets.adobedtm.com/5d4962a43b79/c1061d2c5e7b/launch-191c2462b890.min.js" ></script>
    """
    },
    "extra_footer": {
        """
    <script type="text/javascript">if (typeof _satellite !== "undefined") {_satellite.pageBottom();}</script>
    """
    },
}


# Add any paths that contain custom static files (such as style sheets) here,
html_css_files = ["swagger-nvidia.css"]

# These folders are copied to the documentation's HTML output
html_static_path = ['css']

# Include these files in the root of the built documentation
html_extra_path = ["project.json"]
