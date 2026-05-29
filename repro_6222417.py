"""
Reproduction script for NVBug 6222417 ([RAG BP][v2.6.0][RC2] Getting error in
rag library lite notebook).

Mirrors the lite-mode flow of notebooks/rag_library_lite_usage.ipynb:
  1. Configure NV-Ingest cloud endpoints + NGC_API_KEY
  2. Start NV-Ingest subprocess
  3. Create NvidiaRAGIngestor in lite mode pointed at a fresh milvus-lite db
  4. Run create_collection (original NVBug failure point, fixed via PR #625)
  5. Run upload_documents (REOPENED NVBug failure point — Renu's "different
     issue while adding documents")
  6. Poll status until done; print any error
"""

import asyncio
import logging
import os
import sys
import time
import traceback
from pathlib import Path

# --- Configure logging --------------------------------------------------------
LOG_PATH = "/tmp/repro_6222417.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_PATH, mode="w"),
    ],
)
log = logging.getLogger("repro_6222417")

# --- Environment setup --------------------------------------------------------
# NGC_API_KEY is required by the package; mirror from NVIDIA_API_KEY if needed.
if not os.environ.get("NGC_API_KEY"):
    if os.environ.get("NVIDIA_API_KEY"):
        os.environ["NGC_API_KEY"] = os.environ["NVIDIA_API_KEY"]
    else:
        log.error("Neither NGC_API_KEY nor NVIDIA_API_KEY set in env")
        sys.exit(2)

# Mirror cell 11 of the notebook — NV-Ingest cloud endpoints (used by the
# nv-ingest subprocess for OCR / page / graphic / table extraction).
os.environ.setdefault("OCR_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-ocr-v1")
os.environ.setdefault("OCR_INFER_PROTOCOL", "http")
os.environ.setdefault("YOLOX_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-page-elements-v3")
os.environ.setdefault("YOLOX_INFER_PROTOCOL", "http")
os.environ.setdefault("YOLOX_GRAPHIC_ELEMENTS_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-graphic-elements-v1")
os.environ.setdefault("YOLOX_GRAPHIC_ELEMENTS_INFER_PROTOCOL", "http")
os.environ.setdefault("YOLOX_TABLE_STRUCTURE_HTTP_ENDPOINT", "https://ai.api.nvidia.com/v1/cv/nvidia/nemotron-table-structure-v1")
os.environ.setdefault("YOLOX_TABLE_STRUCTURE_INFER_PROTOCOL", "http")

# Workaround for the ORIGINAL create_collection bug (PR #625 not in this
# bugfix branch): use a fresh, unique milvus-lite path so the create succeeds.
# That lets us reach the REOPENED add-documents failure point.
REPRO_DB_PATH = f"/tmp/repro_6222417_milvus-{int(time.time())}.db"
log.info("Using milvus-lite db path: %s", REPRO_DB_PATH)

# Files to ingest (same as notebook cell 26, but with paths from repo root).
INGEST_FILES = [
    "data/multimodal/woods_frost.docx",
    "data/multimodal/multimodal_test.pdf",
]
for f in INGEST_FILES:
    if not Path(f).is_file():
        log.error("Input file missing: %s", f)
        sys.exit(3)
    log.info("Input file present: %s (%d bytes)", f, Path(f).stat().st_size)

# --- Start NV-Ingest pipeline subprocess -------------------------------------
log.info("Starting NV-Ingest subprocess (port 7671)...")
from nv_ingest.framework.orchestration.ray.util.pipeline.pipeline_runners import run_pipeline  # noqa: E402

# port override: notebook uses 7671 (lite/library mode) vs 7670 (server mode)
os.environ.setdefault("INGEST_INTAKE_PORT", "7671")
try:
    run_pipeline(block=False, disable_dynamic_scaling=True, run_in_subprocess=True)
    log.info("NV-Ingest subprocess launched")
except Exception:
    log.exception("Failed to start NV-Ingest subprocess")
    raise

# Give NV-Ingest a moment to bind its ports
time.sleep(15)


# --- Notebook cell 20 — Initialize NvidiaRAGIngestor in lite mode ------------
from nvidia_rag import NvidiaRAGIngestor  # noqa: E402
from nvidia_rag.utils.configuration import NvidiaRAGConfig  # noqa: E402

log.info("Loading config from notebooks/config.yaml")
config_ingestor = NvidiaRAGConfig.from_yaml("notebooks/config.yaml")
config_ingestor.vector_store.name = "milvus"
config_ingestor.vector_store.url = REPRO_DB_PATH  # workaround for PR #625
config_ingestor.nv_ingest.message_client_port = 7671
config_ingestor.embeddings.server_url = "https://integrate.api.nvidia.com/v1"

log.info("Constructing NvidiaRAGIngestor(mode='lite')")
ingestor = NvidiaRAGIngestor(config=config_ingestor, mode="lite")


async def main() -> int:
    # --- Cell 22 — Create a new collection -----------------------------------
    log.info("=== STEP: create_collection ===")
    try:
        resp = ingestor.create_collection(collection_name="test_library_repro")
        log.info("create_collection response: %r", resp)
    except Exception:
        log.exception("create_collection raised")
        return 22

    # --- Cell 24 — List collections (sanity check) ----------------------------
    log.info("=== STEP: get_collections ===")
    try:
        resp = ingestor.get_collections()
        log.info("get_collections response: %r", resp)
    except Exception:
        log.exception("get_collections raised")
        return 24

    # --- Cell 26 — upload_documents (REOPENED bug failure point) -------------
    log.info("=== STEP: upload_documents (THE REOPENED FAILURE POINT) ===")
    try:
        resp = await ingestor.upload_documents(
            collection_name="test_library_repro",
            blocking=False,
            split_options={"chunk_size": 512, "chunk_overlap": 150},
            filepaths=INGEST_FILES,
        )
        log.info("upload_documents response: %r", resp)
    except Exception:
        log.exception("upload_documents raised — POTENTIAL REOPENED BUG SIGNAL")
        return 26

    task_id = resp.get("task_id") if isinstance(resp, dict) else None
    if not task_id:
        log.error("No task_id in upload_documents response: %r", resp)
        return 26

    # --- Cell 28 — Poll status -----------------------------------------------
    log.info("=== STEP: status polling (task_id=%s) ===", task_id)
    deadline = time.time() + 600
    last = None
    while time.time() < deadline:
        try:
            st = await ingestor.status(task_id=task_id)
        except Exception:
            log.exception("status raised")
            return 28
        if st != last:
            log.info("status: %r", st)
            last = st
        state = (st or {}).get("state") or (st or {}).get("status")
        if state in {"PENDING", "PROGRESS", "STARTED", "RUNNING", None}:
            await asyncio.sleep(5)
            continue
        if state in {"SUCCESS", "FINISHED", "COMPLETED"}:
            log.info("Ingestion finished successfully")
            return 0
        # FAILURE / ERROR / UNKNOWN
        log.error("Terminal non-success status: %r", st)
        return 28

    log.error("status polling timed out after 600s, last=%r", last)
    return 99


if __name__ == "__main__":
    try:
        rc = asyncio.run(main())
    except SystemExit:
        raise
    except Exception:
        log.exception("UNHANDLED in main()")
        rc = 1
    log.info("EXIT_CODE=%d", rc)
    sys.exit(rc)
