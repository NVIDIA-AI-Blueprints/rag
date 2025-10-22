<!--
  SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
  SPDX-License-Identifier: Apache-2.0
-->
:orphan:
# Documentation Development

- [Documentation Development](#documentation-development)
  - [Build the Documentation](#build-the-documentation)
  - [Live Building](#live-building)

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