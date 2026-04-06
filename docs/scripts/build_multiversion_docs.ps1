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

<#
.SYNOPSIS
  Build Sphinx HTML for multiple release lines into a single publish layout.

.DESCRIPTION
  For each version, checks out git ref v{version} (tag) or release-v{version} (branch),
  writes the canonical docs/versions1.json (so every build lists the same versions),
  runs verify_doc_version_manifest.py, then sphinx-build into
  docs/_build/multiversion/{version}/.

  Copies the same manifest to docs/_build/multiversion/versions1.json for the
  version switcher when the site root is this folder.

  Requires a clean working tree unless -AllowDirty is used.

.PARAMETER Versions
  Semver strings without a leading v, e.g. 2.3.0, 2.4.0, 2.5.0

.PARAMETER CanonicalManifest
  Path to the versions1.json to inject on every checkout (default: docs/versions1.json
  from the working tree at script start — save a backup first if needed).

.PARAMETER OutputRoot
  Directory under docs/ that will contain per-version folders and root versions1.json
  (default: docs/_build/multiversion).

.EXAMPLE
  .\docs\scripts\build_multiversion_docs.ps1 -Versions @('2.3.0','2.4.0','2.5.0')

.EXAMPLE
  .\docs\scripts\build_multiversion_docs.ps1 -DryRun
#>

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [string[]]$Versions = @('2.3.0', '2.4.0', '2.5.0'),

    [string]$CanonicalManifest = '',

    [string]$OutputRoot = '',

    [switch]$DryRun,

    [switch]$AllowDirty
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot '..\..')).Path
Set-Location $RepoRoot

if (-not $OutputRoot) {
    $OutputRoot = Join-Path $RepoRoot 'docs\_build\multiversion'
} else {
    $OutputRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutputRoot)
}

if (-not $CanonicalManifest) {
    $CanonicalManifest = Join-Path $RepoRoot 'docs\versions1.json'
} else {
    $CanonicalManifest = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($CanonicalManifest)
}

function Resolve-VersionGitRef {
    param([string]$Version)
    $tag = "v$Version"
    git rev-parse -q --verify "refs/tags/$tag" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return $tag
    }
    $branch = "release-v$Version"
    git rev-parse -q --verify "refs/heads/$branch" 2>$null | Out-Null
    if ($LASTEXITCODE -eq 0) {
        return $branch
    }
    throw "No git tag '$tag' or branch '$branch' found for version $Version"
}

if (-not $DryRun) {
    $dirty = git status --porcelain
    if ($dirty -and -not $AllowDirty) {
        throw "Working tree is dirty. Commit or stash changes, or pass -AllowDirty."
    }
}

if (-not (Test-Path -LiteralPath $CanonicalManifest)) {
    throw "Canonical manifest not found: $CanonicalManifest"
}

$canonicalJson = [System.IO.File]::ReadAllText($CanonicalManifest)

$origHead = git rev-parse HEAD

try {
    if (-not $DryRun) {
        New-Item -ItemType Directory -Force -Path $OutputRoot | Out-Null
    }

    foreach ($ver in $Versions) {
        $ref = Resolve-VersionGitRef -Version $ver
        $dest = Join-Path $OutputRoot $ver

        Write-Host "==> Version $ver <= ref $ref => $dest" -ForegroundColor Cyan

        if ($DryRun) {
            continue
        }

        git checkout $ref
        [System.IO.File]::WriteAllText(
            (Join-Path $RepoRoot 'docs\versions1.json'),
            $canonicalJson,
            [System.Text.UTF8Encoding]::new($false)
        )

        & uv run python docs/scripts/verify_doc_version_manifest.py
        if ($LASTEXITCODE -ne 0) {
            throw "verify_doc_version_manifest.py failed for $ver (ref $ref)"
        }

        if (Test-Path -LiteralPath $dest) {
            Remove-Item -LiteralPath $dest -Recurse -Force
        }
        New-Item -ItemType Directory -Force -Path $dest | Out-Null

        & uv run --group docs sphinx-build docs $dest
        if ($LASTEXITCODE -ne 0) {
            throw "sphinx-build failed for $ver"
        }
    }

    if (-not $DryRun) {
        $rootManifest = Join-Path $OutputRoot 'versions1.json'
        [System.IO.File]::WriteAllText(
            $rootManifest,
            $canonicalJson,
            [System.Text.UTF8Encoding]::new($false)
        )
        Write-Host "Wrote $rootManifest" -ForegroundColor Green
    }
}
finally {
    if (-not $DryRun) {
        git checkout $origHead
        Write-Host "Restored HEAD to $origHead" -ForegroundColor DarkGray
    }
}

Write-Host 'Done.' -ForegroundColor Green
