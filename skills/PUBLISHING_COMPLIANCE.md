# Publishing Compliance — NVIDIA RAG Blueprint

Tracking doc for compliance with the [Skills Publishing Onboarding Guide](https://docs.google.com/document/d/1SNFRQCv0_p3DC2a_IWIf0cB3c49tSE1d/) (owner: Moshe Abramovitch, `moshea@nvidia.com`).

The catalog at `github.com/nvidia/skills` mirrors this repo via `components.d/rag-blueprint.yml`. This document is the single source of truth for what is repo-local-complete vs. what needs cross-team coordination.

> ## ⚠ Hard deadline — **2026-05-27 (Wed)** — get skills signed for Computex
>
> Per Moshe Abramovitch's 2026-05-22 email "Action needed by 2026-05-27 — get your skills signed for the NVIDIA Verified catalog":
> **After 2026-05-27, unsigned skills are no longer eligible for catalog sync.**
>
> Signing runs through NVCARPS (not via this repo). One-time setup, then signing piggybacks on the normal review flow.
>
> Onboarding path for this repo (GitHub-canonical at `NVIDIA-AI-Blueprints/rag`):
> [GitHub-First — Outbound Repos Onboarding](https://nvidia.atlassian.net/wiki/spaces/GAIT/pages/3483240468)
>
> What you get after signing:
> - `skill.oms.sig` per skill (cryptographic signature)
> - `SKILLCARD.yaml` or `skill-card.md` (trust manifest)
> - `BENCHMARK.md` — explicitly not required for the 2026-05-27 cutover (we've shipped stubs anyway)
>
> Skill names are **grandfathered** before this taxonomy guidance — our `rag-blueprint`, `rag-eval`, `rag-perf` need no rename.
>
> PICs: Yashraj Basavaraj Patil (NVCARPS pipeline backup — Mohit OOO 5/25–5/29), Sayali Kandarkar + Moshe Abramovitch (primary). Slack `#nv-carps-support`, email `AgentSkills_Help@exchange.nvidia.com`.

---

## Status

| # | Step (per guide) | Status |
|---|------------------|--------|
| 1 | Develop the skill (SKILL.md per agentskills.io spec, canonical `skills/` path) | ✅ Complete |
| 2 | Self-check with NV-BASE (Tier 1) + functional eval (Tier 3) + `BENCHMARK.md` | ⚠ Tier 3 live; Tier 1 disabled (runner install pending); BENCHMARK.md shipped |
| 3 | Licensing / OSRB | ✅ Apache-2.0; OSRB clearance inherited via existing `NVIDIA-AI-Blueprints/rag` clearance |
| 4 | Publish to product repo | ✅ Canonical layout: `skills/{rag-blueprint,rag-eval,rag-perf}/` |
| 5 | Register in `components.d/<slug>.yml` | ⚠ Registered, but path field needs update — see "Catalog PR" below |
| 6 | Verify skills are live | ⚠ Only `rag-blueprint` mirrored on `main`; `rag-eval` + `rag-perf` pending develop→main merge |
| ⚠ | **NVCARPS signing onboarding (Computex cutover 2026-05-27)** | 🔴 Not started — see Pending #1 |

---

## Repo-local — done

- **Canonical path migration.** Skills moved from the legacy `skill-source/.agents/skills/<skill>/` to `skills/<skill>/` at the repo root. The `skill-source/` directory was removed entirely after audit confirmed no internal or external tool hard-codes it.
- **`BENCHMARK.md`** added to all three skills documenting the Harbor-based Tier 3 methodology, evaluator stack, and result tables. Without-skill baseline rows are marked `TODO` pending NV-ACES integration.
- **Cross-skill README** lives at `skills/README.md` (architecture, install, supported deployment modes, notebook integration table).
- **Validator** moved to `scripts/validate_skill_versions.py` with default `--skills-dir skills/`. Unit test at `tests/unit/test_skills/test_api_version_validation.py`.
- **Internal references updated** across `CLAUDE.md`, `AGENTS.md`, `README.md`, `ci/run_skill_eval.sh`, `skill-eval/{README.md,CLAUDE.md,adapters/}`, `.openclaw/`, `.github/skill-eval/AGENTS.md`, `.github/workflows/skills-eval.yml`, `docs/release-notes.md`.
- **NV-BASE workflow comment** in `.github/workflows/skills-nv-base.yml` now points to this doc for the re-enable runbook.

---

## Pending external coordination

> Priority order is by **deadline**, not by the original guide's step number.

### 1. ⚠ NVCARPS signing onboarding — **due 2026-05-27**

**Owner:** repo maintainer (Pranjal).
**Blocks:** catalog sync after 2026-05-27. Unsigned skills get dropped from `github.com/NVIDIA/skills`.

What to do:
1. Open the [GitHub-First — Outbound Repos Onboarding](https://nvidia.atlassian.net/wiki/spaces/GAIT/pages/3483240468) Confluence page (we're a GitHub-canonical repo at `NVIDIA-AI-Blueprints/rag`).
2. Follow the "one-time setup" to wire `NVIDIA-AI-Blueprints/rag` to NVCARPS.
3. Open the PR the page asks for; NVCARPS scans → validates → signs → pushes a signature commit back with `skill.oms.sig` per skill and a `SKILLCARD.yaml` (or `skill-card.md`) trust manifest.
4. Verify the signed artifacts land under each of `skills/rag-blueprint/`, `skills/rag-eval/`, `skills/rag-perf/`.

Contacts if blocked:
- Yashraj Basavaraj Patil (`yashrajbasav@nvidia.com`) — NVCARPS pipeline backup PIC (Mohit Gupta OOO 2026-05-25 → 2026-05-29).
- Sayali Kandarkar (`skandarkar@nvidia.com`) + Moshe Abramovitch (`moshea@nvidia.com`) — primary rollout PICs.
- Slack: `#nv-carps-support` · Email: `AgentSkills_Help@exchange.nvidia.com`.

Reassurance from the email:
- **Names are grandfathered.** `rag-blueprint`, `rag-eval`, `rag-perf` (all product-scoped already) do **not** need to be renamed despite the new taxonomy guidance.
- **`BENCHMARK.md` is not gating the cutover** — "most teams don't have one yet, no action needed for the 2026-05-27 cutover." Our stubs are bonus.

### 2. Catalog PR — update `components.d/rag-blueprint.yml`

**Owner:** repo maintainer (PR target: `NVIDIA/skills`).
**Blocks:** the canonical path won't be enforced by the catalog gate until this lands.

Drafted change for `github.com/NVIDIA/skills:components.d/rag-blueprint.yml`:

```diff
 name: RAG Blueprint
 repo: NVIDIA-AI-Blueprints/rag
 description: RAG pipeline — deploy, configure, troubleshoot, and manage retrieval augmented generation with Docker Compose or Helm.
 skills:
-  - path: skill-source/.agents/skills/
+  - path: skills/
   catalog_dir: rag
```

Drafted PR title: `rag-blueprint: migrate skills path to canonical skills/`

Drafted PR body:

> Migrate the RAG Blueprint skills path from the legacy `skill-source/.agents/skills/` to the canonical `skills/` directory, per the Skills Publishing Onboarding Guide (Step 4 — "Recommended Repository Layout"). The `skill-source/` directory has been removed entirely in the source repo; no skill content is lost.
>
> Companion change in source repo: `NVIDIA-AI-Blueprints/rag` PR #<TODO> on `develop` branch (will be visible in catalog on next merge to `main`).

DCO sign-off required (per the publishing guide and `NVIDIA/skills:README.md`).

### 3. OSRB confirmation for `rag-eval` and `rag-perf`

**Owner:** repo maintainer (escalate to `AgentSkills_Help` / Bernd Weber if any 3rd-party gap is found).
**Blocks:** nothing if the boxes below check out — guide says "Adding skill to an existing OSRB-approved repo (no new 3rd-party dependencies, no IP changes)" requires no OSRB action.

Quick checklist before next release:

- [ ] `rag-eval` introduces no new 3rd-party dep beyond `scripts/eval/pyproject.toml` (verify: RAGAS already cleared upstream).
- [ ] `rag-perf` introduces no new 3rd-party dep beyond `scripts/rag-perf/pyproject.toml` (verify: `aiperf` already cleared upstream).
- [ ] [IP Review Process](https://nvidia.atlassian.net/wiki/spaces/OSS/pages/2529034695/IP+Review+Process) 6-question checklist filled for each skill's most-recent commit. All six answers affirmative.
- [ ] No external (non-`@nvidia.com`) contributor commits in the skill directories. `git log --format='%ae' skills/ | sort -u` should return only `@nvidia.com` addresses.

If any box fails: file an OSRB contribution-request bug — https://nvbugspro.nvidia.com/bug/5682016 — and loop in Bernd Weber per the guide.

### 4. NV-BASE runner — install `nv-base` on `rag-skill-validator`

**Owner:** runner admin (request via `AgentSkills_Help` DL or NVCARPS team).
**Blocks:** Tier 1 schema/security/PII/naming/frontmatter checks. `.github/workflows/skills-nv-base.yml` currently has `if: false`.

Steps to enable once the runner has nv-base:

1. Runner admin installs nv-base at `/opt/nvbase-venv/bin/nv-base` with `urm.nvidia.com/artifactory/api/pypi/nv-shared-pypi/simple` index access.
2. Drop `if: false` in `.github/workflows/skills-nv-base.yml` line ~57.
3. Trigger via `workflow_dispatch` against `*` to smoke-test all three skills before merging the toggle.

Reference: NVCARPS [How-To-Contribute-Skills](https://nvidia.atlassian.net/wiki/spaces/GAIT/pages/2992484731/HOW-To-Contribute-Skills).

### 5. NV-ACES `evals.json` schema migration (optional, future)

**Owner:** repo maintainer + Jean-Francois Puget (JFP, evaluation framework).
**Blocks:** nothing — current Harbor-style `eval/<name>.json` specs (`{skills, platforms, resources.platforms, env, expects[]}`) are the format CI runs today. NV-ACES introduces a complementary `evals.json` (`{id, question, expected_skill, ground_truth, expected_behavior}`) for the deterministic `skill_execution` / `skill_efficiency` evaluators.

Action when NV-ACES integration ships:
- Convert each `eval/<name>.json` to NV-ACES `evals.json` (one per skill, alongside the Harbor specs).
- Populate the "Without skill (baseline)" column in each `BENCHMARK.md`.
- Update `.github/workflows/skills-eval.yml` to dispatch both flows.

### 6. Merge `develop` → `main`

**Owner:** release manager.
**Blocks:** catalog still shows 1 mirrored skill (`rag-blueprint`) until this merge lands; the daily sync only mirrors `main`. After merge: `rag-eval` and `rag-perf` will appear in `github.com/nvidia/skills/skills/rag/` within ~24h.

Pre-merge checks:
- All three skills pass `uv run pytest tests/unit/test_skill_source/`.
- `uv run python scripts/validate_skill_versions.py` exits 0.
- All three SKILL.md `version:` fields match `pyproject.toml:project.version`.

---

## Contacts (per the publishing guide)

| Topic | Contact |
|-------|---------|
| OSS External Skills PIC | Moshe Abramovitch — `moshea@nvidia.com` |
| NVIDIA/skills onboarding | Sayali Kandarkar — `skandarkar@nvidia.com` |
| Evaluation framework / Skills PIC | Jean-Francois Puget — `jpuget@nvidia.com` |
| NVCARPS access | Mohit Gupta — `mohgupta@nvidia.com` |
| Skill Card spec | Michael Boone — `mboone@nvidia.com` |
| Front Door / Distribution | Nikhil Swaminathan — `nswami@nvidia.com` |
| General questions | `AgentSkills_Help` DL |
