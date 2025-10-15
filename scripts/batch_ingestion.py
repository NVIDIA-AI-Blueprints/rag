# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Batch Ingestion Script

- Scans a folder and uploads documents to the ingestor-server in sequential batches
- Professional logging
- Shows batch progress (e.g., "Uploading batch 2/43 ...")
- Polls ingestion task until FINISHED, continues with next batch; on FAILED/UNKNOWN or any unexpected state, marks batch as failed

Notes:
- Can optionally create and/or delete collections via CLI flags.
- Its purpose is to ease bulk upload for large datasets.
"""

import argparse
import json
import logging
import math
import os
import sys
import time
from collections.abc import Iterable
from pathlib import Path

import requests

DEFAULT_PORT = 8082
POLL_INTERVAL = 5  # seconds
POLL_TIMEOUT = 60 * 60 * 6  # 6 hours


def configure_logger(verbosity: int) -> logging.Logger:
    level = logging.WARNING
    if verbosity == 1:
        level = logging.INFO
    elif verbosity >= 2:
        level = logging.DEBUG

    logger = logging.getLogger("batch_ingestion")
    logger.setLevel(level)
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    if logger.handlers:
        logger.handlers.clear()
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def chunked(iterable: list[Path], chunk_size: int) -> Iterable[list[Path]]:
    for i in range(0, len(iterable), chunk_size):
        yield iterable[i : i + chunk_size]


def discover_files(folder: Path, allowed_exts: tuple[str, ...]) -> list[Path]:
    files: list[Path] = []
    for root, _dirs, filenames in os.walk(folder):
        for name in filenames:
            p = Path(root) / name
            if not allowed_exts or p.suffix.lower() in allowed_exts:
                files.append(p)
    files.sort()
    return files


def form_documents_payload(
    collection_name: str,
    blocking: bool,
    split_chunk_size: int,
    split_chunk_overlap: int,
    generate_summary: bool,
    custom_metadata: list[dict] | None = None,
):
    return {
        "collection_name": collection_name,
        "blocking": blocking,
        "split_options": {
            "chunk_size": split_chunk_size,
            "chunk_overlap": split_chunk_overlap,
        },
        "custom_metadata": custom_metadata or [],
        "generate_summary": generate_summary,
    }


class IngestionClient:
    def __init__(
        self,
        base_url: str,
        logger: logging.Logger,
        poll_timeout: int | float | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self.logger = logger
        self.poll_timeout = poll_timeout if poll_timeout is not None else POLL_TIMEOUT

    def create_collection(
        self,
        collection_name: str,
        embedding_dimension: int = 2048,
        metadata_schema: list[dict] | None = None,
    ) -> dict:
        """Create a collection using POST /v1/collection."""
        url = f"{self.base_url}/v1/collection"
        payload = {
            "collection_name": collection_name,
            "embedding_dimension": embedding_dimension,
            "metadata_schema": metadata_schema or [],
        }
        try:
            resp = requests.post(url, json=payload, timeout=60)
            if resp.status_code >= 400:
                try:
                    self.logger.error(
                        f"Create collection failed {resp.status_code}: {resp.text}"
                    )
                except Exception:
                    pass
                resp.raise_for_status()
            return resp.json() if resp.text else {"status": "ok"}
        except Exception as e:
            raise RuntimeError(f"Failed to create collection: {e}") from e

    def delete_collections(self, collection_names: list[str]) -> dict:
        """Delete collections using DELETE /v1/collections with JSON body [names]."""
        url = f"{self.base_url}/v1/collections"
        try:
            resp = requests.delete(url, json=collection_names, timeout=60)
            if resp.status_code >= 400:
                try:
                    self.logger.error(
                        f"Delete collections failed {resp.status_code}: {resp.text}"
                    )
                except Exception:
                    pass
                resp.raise_for_status()
            return resp.json() if resp.text else {"status": "ok"}
        except Exception as e:
            raise RuntimeError(f"Failed to delete collections: {e}") from e

    def list_documents(self, collection_name: str) -> list[str]:
        """Return list of existing document names (filenames) for the collection."""
        url = f"{self.base_url}/v1/documents"
        params = {"collection_name": collection_name}
        try:
            resp = requests.get(url, params=params, timeout=60)
            if resp.status_code >= 400:
                try:
                    self.logger.error(
                        f"List documents failed {resp.status_code}: {resp.text}"
                    )
                except Exception:
                    pass
                resp.raise_for_status()
            data = resp.json() or {}
            docs = data.get("documents", [])
            names = []
            for d in docs:
                # Prefer metadata.filename if present, else document_name
                meta = d.get("metadata") or {}
                filename = meta.get("filename") or d.get("document_name")
                if filename:
                    names.append(filename)
            return names
        except Exception as e:
            raise RuntimeError(f"Failed to list documents: {e}") from e

    def upload_documents(self, files: list[Path], payload: dict) -> dict:
        url = f"{self.base_url}/v1/documents"
        files_form = []
        try:
            for p in files:
                content_type = _guess_content_type(p)
                files_form.append(
                    (
                        "documents",
                        (p.name, open(p, "rb"), content_type),
                    )
                )
            # Add the JSON payload as a multipart field named "data" with application/json
            files_form.append(
                (
                    "data",
                    (None, json.dumps(payload), "application/json"),
                )
            )

            response = requests.post(url, files=files_form, timeout=300)
            if response.status_code >= 400:
                try:
                    err_text = response.text
                except Exception:
                    err_text = "<no response text>"
                self.logger.error(
                    f"Upload request failed with {response.status_code}: {err_text}"
                )
                response.raise_for_status()
            return response.json()
        finally:
            for _field_name, file_tuple in files_form:
                # file_tuple = (filename, fileobj, content_type)
                try:
                    file_tuple[1].close()
                except Exception:
                    pass

    def poll_task_status(self, task_id: str) -> dict:
        url = f"{self.base_url}/v1/status"
        params = {"task_id": task_id}
        start_time = time.time()
        self.logger.info(f"    - ⏳ Polling task status for task_id: {task_id}")
        poll_success = True
        retries = 1
        spinner_frames = ["|", "/", "-", "\\"]
        spinner_idx = 0
        spinner_enabled = sys.stdout.isatty()
        last_spinner_len = 0

        def draw_spinner():
            nonlocal spinner_idx, last_spinner_len
            if not spinner_enabled:
                return
            frame = spinner_frames[spinner_idx]
            spinner_idx = (spinner_idx + 1) % len(spinner_frames)
            msg = f"    - ⏳ Polling task status for task_id: {task_id} {frame}"
            last_spinner_len = len(msg)
            try:
                sys.stdout.write("\r" + msg)
                sys.stdout.flush()
            except Exception:
                pass

        def clear_spinner_line():
            nonlocal last_spinner_len
            if not spinner_enabled or last_spinner_len == 0:
                return
            try:
                sys.stdout.write("\r" + (" " * last_spinner_len) + "\r")
                sys.stdout.flush()
            except Exception:
                pass

        def sleep_with_spinner(seconds: float):
            if not spinner_enabled:
                time.sleep(seconds)
                return
            step = 0.2
            remaining = float(seconds)
            while remaining > 0:
                draw_spinner()
                t = step if remaining > step else remaining
                time.sleep(t)
                remaining -= t

        while True:
            if poll_success:
                retries = 1
            else:
                retries += 1
            try:
                draw_spinner()
                response = requests.get(url, params=params, timeout=60)
                poll_success = True
            except Exception as e:
                clear_spinner_line()
                self.logger.warning(
                    f"    - Error polling task status on url: {url} params: {params} | retry #{retries} | Error: {e}"
                )
                poll_success = False
                if retries > 10:
                    clear_spinner_line()
                    raise RuntimeError(f"Status polling retries exceeded: {e}") from e
                sleep_with_spinner(POLL_INTERVAL)
                continue

            elapsed = time.time() - start_time
            try:
                status_json = response.json()
            except Exception:
                status_json = {"state": "UNKNOWN", "raw": response.text}

            state = (status_json or {}).get("state")
            if int(elapsed) % 600 < POLL_INTERVAL:
                clear_spinner_line()
                self.logger.info(f"    - Task status after {elapsed:.0f}s: {state}")

            if state == "FINISHED":
                clear_spinner_line()
                self.logger.info("    - ✅ Task finished")
                if status_json.get("result", {}).get("failed_documents"):
                    self.logger.error(
                        f"    - Task failed for {len(status_json.get('result', {}).get('failed_documents'))} documents"
                    )
                    self.logger.error(
                        f"    - Failed documents: {json.dumps(status_json.get('result', {}).get('failed_documents'), indent=2)}"
                    )
                return status_json
            if state == "FAILED":
                clear_spinner_line()
                self.logger.error(f"    - Task failed ❌ | {status_json}")
                raise RuntimeError(f"Task failed: {status_json}")
            if state == "UNKNOWN":
                clear_spinner_line()
                self.logger.error(
                    f"    - Task unknown ❌ | Ingestor Server may have restarted | {status_json}"
                )
                raise RuntimeError(f"Task unknown: {status_json}")

            if time.time() - start_time > self.poll_timeout:
                clear_spinner_line()
                self.logger.error(f"    - Task timed out after {self.poll_timeout}s")
                raise TimeoutError("Status polling timed out")

            sleep_with_spinner(POLL_INTERVAL)


def _guess_content_type(path: Path) -> str:
    ext = path.suffix.lower()
    if ext in {".pdf"}:
        return "application/pdf"
    if ext in {".docx"}:
        return "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    if ext in {".txt"}:
        return "text/plain"
    if ext in {".md"}:
        return "text/markdown"
    if ext in {".csv"}:
        return "text/csv"
    if ext in {".png"}:
        return "image/png"
    if ext in {".jpg", ".jpeg"}:
        return "image/jpeg"
    return "application/octet-stream"


def _maybe_delete_collection_on_exit(
    client: "IngestionClient",
    collection_name: str,
    logger: logging.Logger,
    enabled: bool,
):
    if not enabled:
        return
    try:
        logger.info(f"Deleting collection '{collection_name}' ...")
        client.delete_collections([collection_name])
        logger.info("✅ Delete collection request completed.")
    except Exception as e:
        logger.error(f"❌ Could not delete collection: {e}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Upload documents to ingestor-server in sequential batches",
    )
    parser.add_argument(
        "--folder", required=True, help="Folder path to scan for documents"
    )
    parser.add_argument(
        "--collection-name", required=True, help="Target collection name"
    )
    parser.add_argument(
        "--create_collection",
        action="store_true",
        help="Create the target collection before uploading (default: False)",
    )
    parser.add_argument(
        "--delete_collection",
        action="store_true",
        help="Delete the target collection and exit unless --create_collection is also set (default: False)",
    )
    parser.add_argument(
        "--ingestor-host",
        default=os.environ.get("INGESTOR_HOST", "localhost"),
        help="Ingestor server host (default: localhost)",
    )
    parser.add_argument(
        "--ingestor-port",
        type=int,
        default=int(os.environ.get("INGESTOR_PORT", DEFAULT_PORT)),
        help=f"Ingestor server port (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--upload-batch-size",
        type=int,
        default=100,
        help="Number of files per batch",
    )
    parser.add_argument(
        "--allowed-exts",
        nargs="*",
        default=[".pdf", ".docx", ".txt", ".md", ".csv", ".png", ".jpg", ".jpeg"],
        help="Filter by file extensions (default: common doc/image types)",
    )
    parser.add_argument(
        "--split-chunk-size",
        type=int,
        default=512,
        help="Splitter chunk size",
    )
    parser.add_argument(
        "--split-chunk-overlap",
        type=int,
        default=150,
        help="Splitter chunk overlap",
    )
    parser.add_argument(
        "--generate-summary",
        action="store_true",
        help="Generate summaries after ingestion",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="count",
        default=1,
        help="Increase verbosity (-v info, -vv debug)",
    )
    parser.add_argument(
        "--poll-timeout",
        type=int,
        default=None,
        help="Override polling timeout in seconds (default: 6 hours)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logger(args.verbose)

    folder = Path(args.folder).expanduser().resolve()
    if not folder.exists() or not folder.is_dir():
        logger.error(f"Folder not found or not a directory: {folder}")
        return 2

    allowed_exts = tuple(
        e.lower() if e.startswith(".") else f".{e.lower()}" for e in args.allowed_exts
    )
    all_files = discover_files(folder, allowed_exts)
    if not all_files:
        logger.warning(
            "No files discovered matching allowed extensions; nothing to upload."
        )
        return 0

    base_url = f"http://{args.ingestor_host}:{args.ingestor_port}"
    poll_timeout = args.poll_timeout if args.poll_timeout is not None else POLL_TIMEOUT
    client = IngestionClient(base_url, logger, poll_timeout=poll_timeout)

    # Create collection at start if requested
    if args.create_collection:
        logger.info(f"Creating collection '{args.collection_name}' ...")
        try:
            client.create_collection(args.collection_name)
            logger.info("✅ Create collection request completed.")
        except Exception as e:
            logger.error(f"❌ Could not create collection: {e}")
            return 5

    # Fetch existing docs and filter
    logger.info(
        f"Fetching existing documents from collection '{args.collection_name}' to skip already ingested files..."
    )
    try:
        existing_names = set(client.list_documents(args.collection_name))
    except Exception as e:
        logger.error(f"Could not fetch existing documents: {e}")
        return 3

    files = [p for p in all_files if p.name not in existing_names]
    skipped = [p for p in all_files if p.name in existing_names]

    if skipped:
        logger.info(
            f"Skipping {len(skipped)} already ingested file(s). Example(s): {[p.name for p in skipped[:10]]}"
        )

    total_files = len(files)
    if total_files == 0:
        logger.info("All files are already ingested. Nothing to upload.")
        return 0

    batches = list(chunked(files, args.upload_batch_size))
    total_batches = len(batches)

    logger.info(
        f"Discovered {total_files} files in {folder}. Uploading in {total_batches} batch(es) of up to {args.upload_batch_size}."
    )

    failures: list[str] = []
    for batch_index, file_batch in enumerate(batches, start=1):
        batch_label = f"batch {batch_index}/{total_batches}"
        logger.info(f"⏳ Uploading {batch_label} with {len(file_batch)} file(s)...")

        payload = form_documents_payload(
            collection_name=args.collection_name,
            blocking=False,
            split_chunk_size=args.split_chunk_size,
            split_chunk_overlap=args.split_chunk_overlap,
            generate_summary=args.generate_summary,
        )

        try:
            response_json = client.upload_documents(file_batch, payload)
        except Exception as e:
            logger.exception(f"Failed to upload {batch_label}: {e}")
            failures.extend([p.name for p in file_batch])
            continue

        task_id = (
            (response_json or {}).get("task_id")
            or (response_json or {}).get("task")
            or (response_json or {}).get("id")
        )
        if not task_id:
            logger.error(
                f"Could not find task id in response for {batch_label}: {response_json}"
            )
            failures.extend([p.name for p in file_batch])
            continue

        logger.info(f"Started ingestion for {batch_label}; task_id={task_id}")

        try:
            client.poll_task_status(task_id)
            logger.info(f"Completed {batch_label} ✅")
        except Exception as e:
            logger.error(f"{batch_label} failed during polling: {e}")
            failures.extend([p.name for p in file_batch])
            continue

    exit_code = 0
    if failures:
        logger.error(
            f"Completed with failures. {len(failures)} file(s) failed across batches. Examples: {failures[:10]}"
        )
        exit_code = 1
    else:
        logger.info("All batches completed successfully. ✅")

    # Delete collection at end if requested
    _maybe_delete_collection_on_exit(
        client, args.collection_name, logger, args.delete_collection
    )

    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
