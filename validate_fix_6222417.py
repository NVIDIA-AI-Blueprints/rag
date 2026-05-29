"""Validation script for the NVBug 6222417 fix.

Runs the patched ``notebooks/rag_library_lite_usage.ipynb`` Cell 14 verbatim
(via exec) so we exercise the fix exactly as a notebook user would. If the
fix is correct, the env var set in Cell 14 will disable Ray's
uv_runtime_env_hook even though this validation script is itself launched
from a ``uv run`` ancestor (the agentic-bugfix harness).

Without the fix: workers fail to import ray, the pipeline never wires up,
docs stay in "submitted" state.

With the fix: workers reuse sys.executable, the pipeline starts, both docs
ingest successfully.
"""

import asyncio
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path

LOG_PATH = "/tmp/validate_fix_6222417.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("validate_fix_6222417")

# --- Sanity: NGC_API_KEY ------------------------------------------------------
if not os.environ.get("NGC_API_KEY"):
    if os.environ.get("NVIDIA_API_KEY"):
        os.environ["NGC_API_KEY"] = os.environ["NVIDIA_API_KEY"]
    else:
        log.error("Neither NGC_API_KEY nor NVIDIA_API_KEY set")
        sys.exit(2)

# --- Exec the notebook's env-block cell (Cell 11) and run_pipeline cell -------
# (Cell 14) verbatim. The fix lives in Cell 11; Cell 14 is unchanged. We
# exec both to match what a real notebook run does.
NB_PATH = Path("notebooks/rag_library_lite_usage.ipynb")
with NB_PATH.open() as f:
    nb = json.load(f)

env_block_cell_source = None
run_pipeline_cell_source = None
for i, cell in enumerate(nb["cells"]):
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", ""))
    if "RAY_ENABLE_UV_RUN_RUNTIME_ENV" in src and env_block_cell_source is None:
        env_block_cell_source = src
        log.info("Located env-block (with Ray hook disable) in notebook cell %d", i)
    if "run_pipeline(" in src and run_pipeline_cell_source is None:
        run_pipeline_cell_source = src
        log.info("Located run_pipeline in notebook cell %d", i)

assert env_block_cell_source is not None, (
    "Fix is NOT applied — RAY_ENABLE_UV_RUN_RUNTIME_ENV is not set in any "
    "notebook cell. This validation cannot confirm the fix."
)
assert run_pipeline_cell_source is not None, "No run_pipeline cell found"
log.info("✓ Pre-condition: env-block cell sets RAY_ENABLE_UV_RUN_RUNTIME_ENV")

pre_value = os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV")
log.info("Pre-exec env var value: %r", pre_value)

log.info("=== EXEC-ing patched Cell 11 (env block) verbatim ===")
exec(compile(env_block_cell_source, "<notebook-env-block>", "exec"), globals())

log.info("=== EXEC-ing Cell 14 (run_pipeline) verbatim ===")
exec(compile(run_pipeline_cell_source, "<notebook-run-pipeline>", "exec"), globals())

# Post-exec assertion
post_value = os.environ.get("RAY_ENABLE_UV_RUN_RUNTIME_ENV")
log.info("Post-exec env var value: %r", post_value)
assert post_value == "0", f"Fix did not take effect; env var = {post_value!r}"
log.info("✓ Fix in effect: RAY_ENABLE_UV_RUN_RUNTIME_ENV=0 after exec")

# Give NV-Ingest a moment to bind ports + start Ray workers
time.sleep(15)


# --- Cell 20 — initialise NvidiaRAGIngestor in lite mode ---------------------
from nvidia_rag import NvidiaRAGIngestor  # noqa: E402
from nvidia_rag.utils.configuration import NvidiaRAGConfig  # noqa: E402

REPRO_DB_PATH = f"/tmp/validate_fix_6222417_milvus-{int(time.time())}.db"
log.info("Milvus-Lite path: %s", REPRO_DB_PATH)

config_ingestor = NvidiaRAGConfig.from_yaml("notebooks/config.yaml")
config_ingestor.vector_store.name = "milvus"
config_ingestor.vector_store.url = REPRO_DB_PATH
config_ingestor.nv_ingest.message_client_port = 7671
config_ingestor.embeddings.server_url = "https://integrate.api.nvidia.com/v1"
ingestor = NvidiaRAGIngestor(config=config_ingestor, mode="lite")

INGEST_FILES = [
    "data/multimodal/woods_frost.docx",
    "data/multimodal/multimodal_test.pdf",
]


async def main() -> int:
    log.info("=== STEP: create_collection ===")
    resp = ingestor.create_collection(collection_name="test_library_validate")
    log.info("create_collection: %r", resp)

    log.info("=== STEP: upload_documents ===")
    resp = await ingestor.upload_documents(
        collection_name="test_library_validate",
        blocking=False,
        split_options={"chunk_size": 512, "chunk_overlap": 150},
        filepaths=INGEST_FILES,
    )
    log.info("upload_documents: %r", resp)
    task_id = resp.get("task_id")
    if not task_id:
        log.error("No task_id in response")
        return 1

    log.info("=== STEP: status poll ===")
    deadline = time.time() + 600
    last_state = None
    while time.time() < deadline:
        st = await ingestor.status(task_id=task_id)
        state = (st or {}).get("state") or (st or {}).get("status")
        if state != last_state:
            log.info("state: %s | st: %r", state, st)
            last_state = state
        if state in {"SUCCESS", "FINISHED", "COMPLETED"}:
            break
        if state in {"FAILURE", "ERROR"}:
            log.error("Terminal failure status: %r", st)
            return 2
        await asyncio.sleep(5)
    else:
        log.error("Polling timed out at 600s")
        return 3

    # The bug: state may be FINISHED but failed_documents non-empty.
    # Check for the actual ingestion success.
    result = (st or {}).get("result", {})
    failed = result.get("failed_documents", [])
    documents = result.get("documents", [])
    log.info("Final: %d succeeded, %d failed", len(documents), len(failed))
    if failed:
        log.error("Some documents failed: %r", failed)
        return 4
    if len(documents) < 2:
        log.error("Expected 2 documents ingested, got %d", len(documents))
        return 5

    log.info("✓ Validation passed: %d documents ingested cleanly", len(documents))
    return 0


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except Exception:
        log.exception("UNHANDLED in main()")
        rc = 1
    log.info("EXIT_CODE=%d", rc)
    sys.exit(rc)
