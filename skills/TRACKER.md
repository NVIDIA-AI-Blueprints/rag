# RAG Skills and OpenClaw Tracker

Status values: `Not Started`, `In Progress`, `Blocked`, `Done`.

## Phase 1: Inventory

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| Done | Compare VSS skill catalog with RAG skill source | VSS catalog, eval style, references, and install guidance reviewed. |
| Done | Inventory current RAG skill material | `skill-source/.agents/skills/rag-blueprint` mapped to focused skill areas. |
| Done | Inventory RAG docs and deploy paths | Source docs, Docker Compose, Helm, examples, and notebooks identified. |
| In Progress | Identify source-of-truth drift | Each new skill reference lists the docs it depends on. |

## Phase 2: Playbook

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| Done | Add RAG Skills Playbook | `skills/PLAYBOOK.md` defines naming, metadata, security, eval, and review rules. |
| Done | Choose catalog shape | Focused `rag-*` skills selected over the legacy router. |
| In Progress | Define final eval harness | Starter `evals/evals.json` exists; CI adapter decision remains open. |
| Not Started | Add compliance checker integration | Repo CI runs skill syntax, metadata, secret, and eval checks. |

## Phase 3: Skill Catalog

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| Done | Add catalog README | `skills/README.md` lists skills and install guidance. |
| Done | Scaffold focused skills | Initial `rag-*` skill folders, references, and evals exist. |
| Done | Replace legacy router usage | `AGENTS.md`, `CLAUDE.md`, `README.md`, and `skill-source/README.md` point to the new catalog and deprecate the legacy router. |
| Done | Add legacy route ownership map | `skills/README.md` maps old `rag-blueprint` routes to focused skills. |
| Done | Rename broad model config skill | `rag-configure-models` renamed to `rag-configure-infrastructure`. |
| In Progress | Deepen each skill from source docs | Query, guardrails, and MCP references were expanded; remaining references need reviewed examples against current RAG APIs and deployment files. |
| Not Started | Add host install smoke test | Symlink install works for Codex, Claude Code, and universal `.agents`. |

## Phase 4: Evals

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| Done | Add starter eval specs | Each initial skill has at least 3 cases with security assertions. |
| Done | Strengthen behavior-gate assertions | Evals include confirmation, input validation, source-doc, health verification, and safe-refusal assertions. |
| Done | Add compliance checker | `scripts/skill_compliance_check.py` validates skills and OpenClaw manifest locally and in CI. |
| Not Started | Align with VSS skill-eval harness | Decide whether to adapt `.github/skill-eval` from VSS or use NV-BASE directly. |
| Not Started | Add deployment eval environments | Docker, Helm, and library eval targets are declared. |
| Not Started | Add nightly all-skills run | Ambiguous trigger and cross-skill routing issues are detected. |

## Phase 5: OpenClaw

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| Done | Add OpenClaw workspace scaffold | `.openclaw/workspace` documents RAG identity, overview, and manual. |
| Done | Define plugin manifest | `.openclaw/openclaw.plugin.json` declares plugin id, config schema, and skill paths. |
| In Progress | Validate OpenClaw install | Manifest and skill symlinks exist; runtime install still needs validation. |
| Not Started | Validate on Brev | RAG deploy, ingest, query, and troubleshoot workflows run on a Brev GPU instance. |
| Not Started | Validate with NemoClaw | The RAG OpenClaw package routes to the same `rag-*` skills. |
| Not Started | Add Slack workflow | Slack-based query/troubleshoot flow is documented and tested if needed. |

## Phase 6: Review and Publication

| Status | Work Item | Acceptance Criteria |
|---|---|---|
| In Progress | Security review | Runtime data-classification gates, secret handling, destructive approvals, and stronger evals are in place; human security review remains. |
| Not Started | Duplication review | RAG skills do not duplicate VSS or other NVIDIA catalog skills unnecessarily. |
| In Progress | Publish readiness review | PR template and CI gate added; real CODEOWNERS/reviewer teams still need assignment. |
| Not Started | Central catalog sync | NVIDIA skills catalog mirror path is identified and documented. |

## Project Tracker Artifact

An Excel tracker with summary, work items, skill catalog, risks, timeline, and
review findings is generated at
`project-artifacts/trackers/RAG_Skills_Project_Tracker.xlsx`.
