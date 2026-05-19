# RAG Claw — OpenClaw Plugin

NVIDIA RAG Blueprint agent for [OpenClaw](https://github.com/openclaw/openclaw). Provides skills covering the full RAG lifecycle: prerequisites and deployment, configuration, troubleshooting, RAGAS evaluation, and performance benchmarking.

---

## Prerequisites

### OpenClaw host

| Requirement | Notes |
|-------------|-------|
| **Node.js** | **≥ 22.19.0** (required by OpenClaw and this plugin). System Node 20/21 is not supported. |
| **npm** | Bundled with Node; used for global OpenClaw CLI and local plugin build. |
| **nvm** (recommended) | Installs Node in your home directory so `npm install -g` works without `sudo`. |

Install Node 22 with [nvm](https://github.com/nvm-sh/nvm), then use it in every shell:

```bash
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
source ~/.nvm/nvm.sh
nvm install 22
nvm alias default 22
node -v   # v22.x.x
```

### RAG deployment

The following should be in place before deploying RAG containers. The agent checks and guides you through each step via the `rag-blueprint` skill — this is a quick reference.

| Requirement | Notes | Install guide |
|-------------|-------|---------------|
| NVIDIA GPU driver | See `docs/support-matrix.md` for minimum version | [nvidia.com/drivers](https://www.nvidia.com/en-us/drivers/) — reboot after install |
| Docker Engine | Required for Docker Compose deployments | [docs.docker.com/engine/install/ubuntu](https://docs.docker.com/engine/install/ubuntu/) |
| Docker Compose | v2.x | Included with Docker Desktop / Engine |
| NVIDIA Container Toolkit | Required for self-hosted NIMs | [Container Toolkit install guide](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) |
| NGC API key | `NGC_API_KEY` or `NVIDIA_API_KEY` | [org.ngc.nvidia.com/setup/api-keys](https://org.ngc.nvidia.com/setup/api-keys) |

**Post-Docker install:** add your user to the docker group so containers run without sudo:

```bash
sudo usermod -aG docker $USER && newgrp docker
```

---

## 1. Install OpenClaw

Use the nvm Node 22 shell (see above), then install the CLI globally:

```bash
npm install -g openclaw
openclaw --version
```

Run guided setup once (models, API keys, workspace, gateway):

```bash
openclaw onboard
```

For NVIDIA API Catalog models, you can pass `--auth-choice nvidia-api-key` during onboard, or set `NVIDIA_API_KEY` / `NGC_API_KEY` in your environment and configure later with `openclaw configure`.

---

## 2. Install the RAG Claw Plugin

OpenClaw loads **compiled** JavaScript from `dist/`. Build the plugin from the RAG repo before installing.

**From the cloned RAG repo:**

```bash
cd /path/to/rag/.openclaw
npm install
npm run build
openclaw plugins install /path/to/rag/.openclaw/
```

`npm run build` compiles `index.ts` → `dist/index.js` and copies skills from `skill-source/.agents/skills/` into `skills/`.

**From npm (after publishing):**

```bash
openclaw plugins install @nvidia/openclaw-rag
```

Restart the gateway after install so the plugin loads:

```bash
# foreground gateway: stop and run again
openclaw gateway run

# or systemd user service
systemctl --user restart openclaw-gateway
```

On first gateway start, the plugin copies workspace templates (`BOOTSTRAP.md`, `IDENTITY.md`, `SOUL.md`, `AGENTS.md`, `TOOLS.md`) to `~/.openclaw/workspace/`. If Docker is present, it may write a systemd drop-in for gateway Docker socket access — apply it with:

```bash
systemctl --user daemon-reload
systemctl --user restart openclaw-gateway
```

---

## 3. Verify

```bash
openclaw skills list | grep -E "rag-blueprint|rag-eval|rag-perf"
```

Expected output:

```text
rag-blueprint  Deploy, configure, troubleshoot, and manage the NVIDIA RAG Blueprint
rag-eval       Filesystem RAG benchmarks (corpus/, train.json, RAGAS via evaluate_rag.py)
rag-perf       Performance benchmarking (aiperf load tests) for a deployed RAG server
```

---

## 4. Run and interact

Start the **gateway** (required for chat and the web UI):

```bash
# foreground (good for first try)
openclaw gateway run

# or install and start a user service
openclaw gateway install
systemctl --user start openclaw-gateway
```

Add a long agent timeout for RAG deploys and evals (merge into `~/.openclaw/openclaw.json`):

```json
{
  "agents": {
    "defaults": {
      "timeoutSeconds": 3600
    }
  }
}
```

In another terminal, use any of these:

| Interface | Command |
|-----------|---------|
| Terminal chat (gateway) | `openclaw tui --timeout-ms 3600000` |
| Terminal chat (local) | `openclaw chat --timeout-ms 3600000` |
| Web Control UI | `openclaw dashboard` → `http://127.0.0.1:18789/` |
| One-shot turn | `openclaw agent --message "check prerequisites"` |

Prefer **gateway-backed** `openclaw tui` (not `openclaw chat`) for end-to-end RAG Docker deploys: the gateway keeps the run alive while the terminal UI may show `idle` after ~30s without streamed text. If you see _"This response is taking longer than expected"_, wait for the run to finish or send a short follow-up (for example _"status update"_); avoid starting a new deploy request until the prior turn completes. Inside the TUI, `/verbose full` shows tool progress during long shell work.

Check status:

```bash
openclaw status
openclaw health
```

**First session:** start `openclaw tui` (with the gateway running). The BOOTSTRAP flow runs automatically and the agent introduces itself and walks through initial RAG configuration.

Example prompts:

- _"check prerequisites"_
- _"deploy RAG with NVIDIA-hosted NIMs"_
- _"RAG server is unhealthy"_
- _"run RAGAS eval on my benchmark dataset"_

Optional: install the browser skill into the workspace (used for RAG UI automation):

```bash
cd ~/.openclaw/workspace
npx --yes skills add vercel-labs/agent-browser --yes
```

OpenClaw CLI reference: [docs.openclaw.ai/cli](https://docs.openclaw.ai/cli)

---

## Skills Reference

| Skill | Trigger phrases |
|-------|-----------------|
| `rag-blueprint` | "deploy RAG", "enable VLM", "configure reranker", "RAG is unhealthy", "stop RAG" |
| `rag-eval` | "run RAGAS eval", "evaluate_rag", "benchmark dataset", "parse eval results" |
| `rag-perf` | "rag perf", "aiperf", "latency benchmark", "throughput test" |

Skill source of truth: `skill-source/.agents/skills/` in this repository.

---

## Default service ports

| Service | Port |
|---------|------|
| RAG server API | 8081 |
| Ingestor API | 8082 |
| Web UI | 8090 |

See `docs/service-port-gpu-reference.md` for the full port and GPU map.
