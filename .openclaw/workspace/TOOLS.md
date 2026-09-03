# TOOLS.md - Local Notes

Skills define _how_ tools work. This file is for _your_ specifics — the stuff that's unique to your setup.

Run `BOOTSTRAP.md` on first session to populate the RAG section below automatically.

---

## RAG (NVIDIA RAG Blueprint)

<!-- BOOTSTRAP fills this section -->

- **Repo:** _(run BOOTSTRAP to configure — or fill in your repo path)_
- **Deployment:** _(docker self-hosted | docker nvidia-hosted | docker retrieval-only | helm | library — run BOOTSTRAP)_
- **Config file:** _(deploy/compose/.env | deploy/compose/nvdev.env | values.yaml | notebooks/config.yaml)_
- **NGC / NVIDIA API key:** set via `export NGC_API_KEY=...` or `export NVIDIA_API_KEY=...` — do not store the value here
- **GPU:** _(from BOOTSTRAP — model and VRAM summary)_

### Service endpoints (defaults)

| Service | URL |
|---------|-----|
| RAG server | http://localhost:8081 |
| Ingestor | http://localhost:8082 |
| Web UI | http://localhost:8090 |
