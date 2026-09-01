#!/usr/bin/env bash
# ============================================================================
# >>> GPU TESTING PATCH (Option B) <<<
# This entire file exists to validate H100×2 self-hosted CI end-to-end
# via the existing dispatcher workflow (no YAML changes on main needed).
# Retire once the dispatcher YAML on main accepts an `eval_profile` input.
# Companion files in this patch:
#   ci/run_skill_eval.sh                            (BREV_INSTANCE auto-set)
#   skills/rag-deploy-blueprint/eval/h100.json      (deploy spec)
# ============================================================================
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Convenience wrapper for GPU dispatches via the existing dispatcher
# workflow (.github/workflows/run-branch-script.yml on main), which only
# accepts {ref, script, runner} inputs and has no way to pass arbitrary
# env vars. Trigger:
#
#   gh workflow run "Run Branch Script" --ref main \
#     -f ref=feat/skill-eval-ci \
#     -f script=ci/run_skill_eval_h100.sh \
#     -f runner=rag-eval
#
# This wrapper exports EVAL_PROFILE=h100 then execs run_skill_eval.sh.
# Inside that script: EVAL_PROFILE matches the h100* case → BREV_INSTANCE
# auto-generated → BrevEnvironment selected → auto-discovery finds
# skills/*/eval/h100.json → run_skill_eval.sh's EXIT trap routes
# `dmz.h100x2.pcie` to `brev delete` after the cooldown.
#
# Retire this wrapper once the dispatcher workflow on main accepts an
# `eval_profile` input — then GPU dispatches go through the canonical
# script with `-f eval_profile=h100`.

set -euo pipefail
export EVAL_PROFILE=h100
exec bash "$(dirname "$0")/run_skill_eval.sh" "$@"
