<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
:orphan:
# Documentation Development

- [Documentation Development](#documentation-development)
  - [Build the Documentation](#build-the-documentation)
  - [Live Building](#live-building)
  - [Documentation Version](#documentation-version)
    - [Publishing multiple versions on the public site](#publishing-multiple-versions-on-the-public-site)
    - [Multi-version build script](#multi-version-build-script)

## Build the Documentation

The following sections describe how to set up and build the documentation.

Switch to the documentation source folder and generate HTML output.

```sh
uv run --group docs sphinx-build . _build/html
```

* The resulting HTML files are generated in a `_build/html` folder that is created under the project `docs/` folder.
* The generated python API docs are placed in `apidocs` under the `docs/` folder.

## Live Building

When writing documentation, it can be helpful to serve the documentation and have it update live while you edit.

To do so, run:

```sh
uv run --group docs sphinx-autobuild . _build/html --port 12345 --host 0.0.0.0
```

Open a web browser and go to `http://${HOST_WHERE_SPHINX_COMMAND_RUN}:12345` to view the output.

## Documentation Version

The three files below control the version switcher. Before you attempt to publish a new version of the documentation, update these files to match the latest version numbers.

* docs/versions1.json
* docs/project.json
* docs/conf.py

Validate the manifest and that `release` matches `project.json` before building:

```sh
uv run python docs/scripts/verify_doc_version_manifest.py
```

### Publishing multiple versions on the public site

Use the **same** `docs/versions1.json` content for every release line you build (list every published version; `preferred` should be `true` only for the default, usually the latest). On each **release branch or tag**, set `release` in `conf.py` and `version` in `project.json` to that line’s version (for example `2.4.0` on the `2.4.x` branch), then build:

```sh
uv run --group docs sphinx-build . _build/html
```

Deploy the HTML so each line lives as a **sibling** folder, for example `2.3.0/`, `2.4.0/`, `2.5.0/`. The theme resolves `../versions1.json` from the version **index** page to a file **next to** those folders (the parent directory). Copy the same `docs/versions1.json` to that parent as `versions1.json` when you publish, or ensure your pipeline deploys it there once per release. If you add a version to the manifest, rebuild (or redeploy) each affected tree and refresh the root `versions1.json`; invalidate CDN cache if the menu still looks stale.

### Multi-version build script

From the repository root, you can build several release lines into one tree: `docs/_build/multiversion/{version}/` plus a root `versions1.json`. The script reads your current `docs/versions1.json` as the canonical manifest, then for each version checks out git tag `v{version}` if it exists, otherwise branch `release-v{version}`, writes that manifest into `docs/versions1.json`, runs the verifier, and runs Sphinx. Your original `HEAD` is restored at the end.

Preview which refs will be used (no git or build):

```powershell
.\docs\scripts\build_multiversion_docs.ps1 -DryRun
```

Full build (requires a clean working tree, or pass `-AllowDirty`):

```powershell
.\docs\scripts\build_multiversion_docs.ps1 -Versions @('2.3.0','2.4.0','2.5.0')
```

On Linux or macOS:

```sh
chmod +x docs/scripts/build_multiversion_docs.sh
./docs/scripts/build_multiversion_docs.sh --dry-run
./docs/scripts/build_multiversion_docs.sh --versions 2.3.0,2.4.0,2.5.0
```

Serve the result locally, for example: `python -m http.server 8080 --directory docs/_build/multiversion` and open `http://localhost:8080/2.5.0/` to confirm the switcher.