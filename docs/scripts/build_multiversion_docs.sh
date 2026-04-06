#!/usr/bin/env bash
# Copyright (c) 2025, NVIDIA CORPORATION.  All rights reserved.
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

# Build Sphinx HTML for multiple release lines into docs/_build/multiversion/.
# See build_multiversion_docs.ps1 for behavior and options.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
cd "${REPO_ROOT}"

DRY_RUN=0
ALLOW_DIRTY=0
VERSIONS=(2.3.0 2.4.0 2.5.0)
CANONICAL_MANIFEST="${REPO_ROOT}/docs/versions1.json"
OUTPUT_ROOT="${REPO_ROOT}/docs/_build/multiversion"

usage() {
  echo "Usage: $0 [--dry-run] [--allow-dirty] [--versions V1,V2,...] [--manifest PATH] [--output-root PATH]" >&2
  exit 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dry-run) DRY_RUN=1; shift ;;
    --allow-dirty) ALLOW_DIRTY=1; shift ;;
    --versions)
      IFS=',' read -r -a VERSIONS <<< "$2"
      shift 2
      ;;
    --manifest) CANONICAL_MANIFEST="$2"; shift 2 ;;
    --output-root) OUTPUT_ROOT="$2"; shift 2 ;;
    -h|--help) usage ;;
    *) echo "Unknown option: $1" >&2; usage ;;
  esac
done

resolve_ref() {
  local ver="$1"
  local tag="v${ver}"
  local branch="release-v${ver}"
  if git rev-parse -q --verify "refs/tags/${tag}" >/dev/null 2>&1; then
    echo "${tag}"
    return
  fi
  if git rev-parse -q --verify "refs/heads/${branch}" >/dev/null 2>&1; then
    echo "${branch}"
    return
  fi
  echo "No git tag ${tag} or branch ${branch} for ${ver}" >&2
  return 1
}

if [[ "${DRY_RUN}" -eq 0 ]]; then
  if [[ -n "$(git status --porcelain)" && "${ALLOW_DIRTY}" -eq 0 ]]; then
    echo "Working tree is dirty. Commit or stash, or pass --allow-dirty." >&2
    exit 1
  fi
fi

if [[ ! -f "${CANONICAL_MANIFEST}" ]]; then
  echo "Canonical manifest not found: ${CANONICAL_MANIFEST}" >&2
  exit 1
fi

canonical_json="$(cat "${CANONICAL_MANIFEST}")"
orig_head="$(git rev-parse HEAD)"

if [[ "${DRY_RUN}" -eq 0 ]]; then
  mkdir -p "${OUTPUT_ROOT}"
  trap 'git checkout "${orig_head}"' EXIT
fi

for ver in "${VERSIONS[@]}"; do
  ref="$(resolve_ref "${ver}")"
  dest="${OUTPUT_ROOT}/${ver}"
  echo "==> Version ${ver} <= ref ${ref} => ${dest}"

  if [[ "${DRY_RUN}" -ne 0 ]]; then
    continue
  fi

  git checkout "${ref}"
  printf '%s' "${canonical_json}" >"${REPO_ROOT}/docs/versions1.json"

  uv run python docs/scripts/verify_doc_version_manifest.py

  rm -rf "${dest}"
  mkdir -p "${dest}"
  uv run --group docs sphinx-build docs "${dest}"
done

if [[ "${DRY_RUN}" -eq 0 ]]; then
  printf '%s' "${canonical_json}" >"${OUTPUT_ROOT}/versions1.json"
  echo "Wrote ${OUTPUT_ROOT}/versions1.json"
fi

echo "Done."
