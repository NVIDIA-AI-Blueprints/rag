#!/usr/bin/env python3
"""
Script to update Milvus VDB metadata with document summaries.

This script:
1. Connects to a Milvus vector database
2. Retrieves summaries for documents from the RAG server's /summary endpoint
3. Updates the content_metadata field for all chunks with their document's summary

Usage:
    python update_metadata_with_summaries.py \
        --collection test \
        --milvus-host localhost \
        --milvus-port 19530 \
        --rag-server http://localhost:8081

Requirements:
    pip install pymilvus requests
"""

import argparse
import json
import logging
import os
import sys
from collections import defaultdict
from typing import Any

import requests
from pymilvus import Collection, connections

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Update Milvus VDB metadata with document summaries"
    )
    parser.add_argument(
        "--collection",
        type=str,
        required=True,
        help="Name of the Milvus collection to update",
    )
    parser.add_argument(
        "--milvus-url",
        type=str,
        default=None,
        help="Milvus server URL (e.g., http://localhost:19530). If provided, overrides --milvus-host and --milvus-port",
    )
    parser.add_argument(
        "--milvus-host",
        type=str,
        default="localhost",
        help="Milvus server host (default: localhost)",
    )
    parser.add_argument(
        "--milvus-port",
        type=int,
        default=19530,
        help="Milvus server port (default: 19530)",
    )
    parser.add_argument(
        "--milvus-user",
        type=str,
        default=None,
        help="Milvus username (optional, for authenticated connections)",
    )
    parser.add_argument(
        "--milvus-password",
        type=str,
        default=None,
        help="Milvus password (optional, for authenticated connections)",
    )
    parser.add_argument(
        "--rag-server",
        type=str,
        default="http://localhost:8081",
        help="RAG server URL (default: http://localhost:8081)",
    )
    parser.add_argument(
        "--ingest-server",
        type=str,
        default="http://localhost:8082",
        help="Ingestion server URL for fetching document list (default: http://localhost:8082)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform a dry run without updating the database",
    )
    parser.add_argument(
        "--summary-field",
        type=str,
        default="document_summary",
        help='Name of the field to add in content_metadata (default: "document_summary")',
    )

    return parser.parse_args()


def parse_milvus_connection(milvus_url: str = None, host: str = "localhost", port: int = 19530) -> tuple[str, int]:
    """Parse Milvus connection parameters from URL or host:port."""
    if milvus_url:
        # Parse URL to extract host and port
        from urllib.parse import urlparse
        parsed = urlparse(milvus_url)
        
        # Handle both http://host:port and host:port formats
        if parsed.netloc:
            # URL format: http://localhost:19530
            host = parsed.hostname or "localhost"
            port = parsed.port or 19530
        elif ":" in milvus_url:
            # host:port format
            parts = milvus_url.split(":")
            host = parts[0]
            port = int(parts[1]) if len(parts) > 1 else 19530
        else:
            # Just host
            host = milvus_url
            port = 19530
    
    return host, port


def connect_to_milvus(host: str, port: int, user: str = None, password: str = None):
    """Connect to Milvus server."""
    try:
        logger.info(f"Connecting to Milvus at {host}:{port}...")
        
        connect_params = {
            "host": host,
            "port": port,
        }
        
        if user and password:
            connect_params["user"] = user
            connect_params["password"] = password
            logger.info("Using authenticated connection")
        
        connections.connect(**connect_params)
        logger.info("✓ Successfully connected to Milvus")
        return True
    except Exception as e:
        logger.error(f"✗ Failed to connect to Milvus: {e}")
        return False


def get_collection_info(collection_name: str) -> tuple[Collection, dict]:
    """Get collection and its schema information."""
    try:
        collection = Collection(collection_name)
        collection.load()
        
        schema_info = {
            "name": collection.name,
            "num_entities": collection.num_entities,
            "schema": collection.schema,
        }
        
        logger.info(f"Collection '{collection_name}' info:")
        logger.info(f"  - Number of entities: {schema_info['num_entities']}")
        logger.info(f"  - Schema: {schema_info['schema']}")
        
        return collection, schema_info
    except Exception as e:
        logger.error(f"✗ Failed to get collection info: {e}")
        raise


def get_chunks_for_file(collection: Collection, file_name: str) -> list[dict]:
    """
    Get all chunks for a specific file from VDB.
    Uses filter to only get chunks for this file.
    
    Args:
        collection: Milvus collection
        file_name: Name of the file (e.g., "document.pdf")
    
    Returns:
        list: List of chunk documents with ALL fields
    """
    try:
        # Use filter expression to get only chunks for this file
        # The source_name contains full path, so we check if it ends with filename
        expr = f'source["source_name"] like "%{file_name}"'
        
        logger.debug(f"  Querying chunks for {file_name}...")
        
        # Fetch ALL fields to preserve everything during upsert
        results = collection.query(
            expr=expr,
            output_fields=["*"],  # Get everything
            limit=16384,
        )
        
        logger.debug(f"  Found {len(results)} chunks for {file_name}")
        return results
        
    except Exception as e:
        logger.error(f"  ✗ Failed to query chunks for {file_name}: {e}")
        return []


def update_file_chunks_with_summary(
    collection: Collection,
    file_name: str,
    summary: str,
    summary_field: str,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    Update all chunks for a specific file with summary.
    
    This function:
    1. Queries VDB for all chunks of this file (with filter)
    2. Fetches ALL fields for each chunk
    3. Updates ONLY the summary field in content_metadata
    4. Upserts back (preserving all other fields)
    
    Args:
        collection: Milvus collection
        file_name: Name of the file
        summary: Summary text to add
        summary_field: Field name in content_metadata
        dry_run: If True, don't actually update
    
    Returns:
        tuple: (successful_updates, failed_updates)
    """
    try:
        # Get all chunks for this file (with ALL fields)
        chunks = get_chunks_for_file(collection, file_name)
        
        if not chunks:
            logger.warning(f"  ⚠ No chunks found for {file_name}")
            return 0, 0
        
        logger.info(f"  Updating {len(chunks)} chunks...")
        
        # Update each chunk
        updated_chunks = []
        for chunk in chunks:
            # Get existing content_metadata or initialize
            content_metadata = chunk.get("content_metadata", {})
            if content_metadata is None:
                content_metadata = {}
            
            # Add/update ONLY the summary field
            content_metadata[summary_field] = summary
            
            # Update the metadata in the chunk
            chunk["content_metadata"] = content_metadata
            
            # Add to update list (chunk has ALL fields preserved)
            updated_chunks.append(chunk)
        
        if dry_run:
            logger.info(f"  [DRY RUN] Would update {len(updated_chunks)} chunks")
            return len(updated_chunks), 0
        else:
            # Upsert all chunks (preserves everything, updates summary)
            collection.upsert(updated_chunks)
            logger.info(f"  ✓ Updated {len(updated_chunks)} chunks")
            return len(updated_chunks), 0
    
    except Exception as e:
        logger.error(f"  ✗ Failed to update chunks for {file_name}: {e}")
        return 0, len(chunks) if chunks else 0


def get_documents_from_api(ingest_server_url: str, collection_name: str) -> list[str]:
    """
    Fetch document list from ingestion server API.
    
    Returns:
        list: List of document file names
    """
    try:
        endpoint = f"{ingest_server_url}/v1/documents"
        params = {"collection_name": collection_name}
        
        logger.info(f"Fetching document list from {endpoint}...")
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            documents = data.get("documents", [])
            file_names = [os.path.basename(doc.get("document_name", "")) for doc in documents]
            logger.info(f"✓ Retrieved {len(file_names)} documents from API")
            return file_names
        else:
            logger.error(f"✗ Failed to get documents: HTTP {response.status_code}")
            return []
    
    except Exception as e:
        logger.error(f"✗ Error fetching documents from API: {e}")
        return []


# Functions removed - now using simplified per-file approach


# Removed - using per-file filtering approach instead


def get_summary_from_rag_server(
    rag_server_url: str, collection_name: str, file_name: str
) -> tuple[str, str]:
    """
    Fetch summary from RAG server's /summary endpoint.
    
    Returns:
        tuple: (summary_text, status) where status is SUCCESS, NOT_FOUND, FAILED, etc.
    """
    try:
        # Construct the endpoint URL
        endpoint = f"{rag_server_url}/v1/summary"
        
        # Query parameters
        params = {
            "collection_name": collection_name,
            "file_name": file_name,
            "blocking": False,  # Non-blocking to get immediate status
            "timeout": 300,
        }
        
        logger.debug(f"Fetching summary for {file_name} from {endpoint}")
        
        # Make GET request
        response = requests.get(endpoint, params=params, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            status = data.get("status", "UNKNOWN")
            summary = data.get("summary", "")
            
            if status == "SUCCESS" and summary:
                logger.info(f"  ✓ Retrieved summary for {file_name} ({len(summary)} chars)")
                return summary, "SUCCESS"
            elif status in ["PENDING", "IN_PROGRESS"]:
                logger.warning(f"  ⚠ Summary for {file_name} is {status}")
                return "", status
            elif status == "NOT_FOUND":
                logger.warning(f"  ⚠ Summary for {file_name} not found")
                return "", "NOT_FOUND"
            elif status == "FAILED":
                error = data.get("error", "Unknown error")
                logger.error(f"  ✗ Summary generation failed for {file_name}: {error}")
                return "", "FAILED"
            else:
                logger.warning(f"  ⚠ Unknown status for {file_name}: {status}")
                return summary, status
        else:
            logger.error(
                f"  ✗ Failed to get summary for {file_name}: "
                f"HTTP {response.status_code} - {response.text}"
            )
            return "", "ERROR"
    
    except requests.exceptions.Timeout:
        logger.error(f"  ✗ Timeout while fetching summary for {file_name}")
        return "", "TIMEOUT"
    except requests.exceptions.ConnectionError:
        logger.error(
            f"  ✗ Connection error while fetching summary for {file_name}. "
            f"Is the RAG server running at {rag_server_url}?"
        )
        return "", "CONNECTION_ERROR"
    except Exception as e:
        logger.error(f"  ✗ Error fetching summary for {file_name}: {e}")
        return "", "ERROR"


def update_chunks_with_summary(
    collection: Collection,
    chunk_pks: list[int],
    summary: str,
    summary_field: str,
    batch_size: int = 100,
    dry_run: bool = False,
) -> tuple[int, int]:
    """
    LEGACY FUNCTION - Kept for backward compatibility.
    Use update_file_chunks_with_summary() for better performance.
    
    Update chunks with summary in their content_metadata.
    
    Returns:
        tuple: (successful_updates, failed_updates)
    """
    successful = 0
    failed = 0
    
    # Process in batches
    for i in range(0, len(chunk_pks), batch_size):
        batch_pks = chunk_pks[i : i + batch_size]
        
        try:
            # Fetch current data for this batch - fetch ALL fields (output_fields=["*"])
            # This gets everything including vectors, which is needed for upsert
            expr = f"pk in {batch_pks}"
            batch_docs = collection.query(
                expr=expr,
                output_fields=["*"],  # Fetch all fields
            )
            
            # Prepare updates - keep everything, only modify content_metadata
            updates = []
            for doc in batch_docs:
                # Get existing content_metadata or initialize empty dict
                content_metadata = doc.get("content_metadata", {})
                if content_metadata is None:
                    content_metadata = {}
                
                # Add or update only the summary field
                content_metadata[summary_field] = summary
                
                # Update the content_metadata in the document
                doc["content_metadata"] = content_metadata
                
                # Use the entire document as-is (all fields preserved)
                updates.append(doc)
            
            if dry_run:
                logger.info(
                    f"  [DRY RUN] Would update batch of {len(updates)} chunks "
                    f"(PKs: {batch_pks[0]}-{batch_pks[-1]})"
                )
                successful += len(updates)
            else:
                # Perform upsert to update the entities
                collection.upsert(updates)
                logger.info(
                    f"  ✓ Updated batch of {len(updates)} chunks "
                    f"(PKs: {batch_pks[0]}-{batch_pks[-1]})"
                )
                successful += len(updates)
        
        except Exception as e:
            logger.error(f"  ✗ Failed to update batch: {e}")
            failed += len(batch_pks)
    
    return successful, failed


def main():
    """Main execution function."""
    args = parse_args()
    
    # Parse Milvus connection parameters
    milvus_host, milvus_port = parse_milvus_connection(
        args.milvus_url, args.milvus_host, args.milvus_port
    )
    
    logger.info("=" * 80)
    logger.info("Milvus VDB Metadata Update Script")
    logger.info("=" * 80)
    logger.info(f"Collection: {args.collection}")
    logger.info(f"Milvus: {milvus_host}:{milvus_port}")
    logger.info(f"Ingest Server: {args.ingest_server}")
    logger.info(f"RAG Server: {args.rag_server}")
    logger.info(f"Summary Field: {args.summary_field}")
    logger.info(f"Dry Run: {args.dry_run}")
    logger.info("=" * 80)
    
    # Step 1: Connect to Milvus
    if not connect_to_milvus(
        milvus_host, milvus_port, args.milvus_user, args.milvus_password
    ):
        sys.exit(1)
    
    # Step 2: Get collection info
    try:
        collection, schema_info = get_collection_info(args.collection)
    except Exception as e:
        logger.error(f"Failed to access collection: {e}")
        sys.exit(1)
    
    if schema_info["num_entities"] == 0:
        logger.warning("Collection is empty. Nothing to update.")
        sys.exit(0)
    
    # Step 3: Get document list from ingestion server API
    file_names = get_documents_from_api(args.ingest_server, args.collection)
    
    if not file_names:
        logger.error("No documents found in collection. Nothing to update.")
        sys.exit(0)
    
    logger.info(f"✓ Found {len(file_names)} documents in collection")
    
    # Step 4: Process each file - fetch summary and update chunks
    logger.info("\n" + "=" * 80)
    logger.info("Processing files: Fetch summaries and update chunks")
    logger.info("=" * 80)
    
    total_files = len(file_names)
    files_with_summary = 0
    files_without_summary = 0
    total_chunks_updated = 0
    total_chunks_failed = 0
    
    for idx, file_name in enumerate(file_names, 1):
        logger.info(f"\n[{idx}/{total_files}] Processing: {file_name}")
        
        # Get summary from RAG server
        summary, status = get_summary_from_rag_server(
            args.rag_server, args.collection, file_name
        )
        
        if status == "SUCCESS" and summary:
            files_with_summary += 1
            
            # Update chunks for this file (per-file query with filter)
            successful, failed = update_file_chunks_with_summary(
                collection=collection,
                file_name=file_name,
                summary=summary,
                summary_field=args.summary_field,
                dry_run=args.dry_run,
            )
            
            total_chunks_updated += successful
            total_chunks_failed += failed
            
            if successful > 0:
                logger.info(f"  ✓ Updated {successful} chunks")
            if failed > 0:
                logger.warning(f"  ⚠ {failed} chunks failed")
        else:
            files_without_summary += 1
            logger.warning(
                f"  ⚠ Skipping file (no summary available, status: {status})"
            )
    
    # Step 5: Summary report
    logger.info("\n" + "=" * 80)
    logger.info("Update Summary")
    logger.info("=" * 80)
    logger.info(f"Total files processed: {total_files}")
    logger.info(f"  - Files with summaries: {files_with_summary}")
    logger.info(f"  - Files without summaries: {files_without_summary}")
    logger.info(f"Total chunks updated: {total_chunks_updated}")
    logger.info(f"Total chunks failed: {total_chunks_failed}")
    
    if args.dry_run:
        logger.info("\n*** DRY RUN MODE - No changes were made ***")
    else:
        logger.info("\n✓ Update completed successfully!")
        logger.info(
            f"\nYou can now query chunks with summaries using the field: "
            f"content_metadata[\"{args.summary_field}\"]"
        )
    
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\nScript interrupted by user. Exiting...")
        sys.exit(130)
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        sys.exit(1)
