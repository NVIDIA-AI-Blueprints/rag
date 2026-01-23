#!/usr/bin/env python3
"""
Quick verification script to check if summaries are present in metadata.

Usage:
    python verify_metadata_summaries.py --collection test
"""

import argparse
import sys
from pymilvus import connections, Collection


def parse_args():
    parser = argparse.ArgumentParser(description="Verify summaries in Milvus metadata")
    parser.add_argument("--collection", required=True, help="Collection name")
    parser.add_argument("--milvus-host", default="localhost", help="Milvus host")
    parser.add_argument("--milvus-port", default=19530, type=int, help="Milvus port")
    parser.add_argument(
        "--summary-field",
        default="document_summary",
        help="Summary field name in content_metadata",
    )
    parser.add_argument(
        "--sample-size", default=3, type=int, help="Number of samples to check"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Connect to Milvus
    print(f"Connecting to Milvus at {args.milvus_host}:{args.milvus_port}...")
    connections.connect(host=args.milvus_host, port=args.milvus_port)

    # Get collection
    coll = Collection(args.collection)
    coll.load()

    # Query sample chunks
    results = coll.query(
        expr="pk >= 0",
        output_fields=["pk", "content_metadata", "text", "source"],
        limit=args.sample_size,
    )

    print(f"\n{'=' * 80}")
    print(f"Metadata Summary Verification - Collection: {args.collection}")
    print(f"{'=' * 80}\n")

    chunks_with_summary = 0
    chunks_without_summary = 0

    for idx, result in enumerate(results, 1):
        pk = result["pk"]
        content_metadata = result.get("content_metadata", {})
        summary = content_metadata.get(args.summary_field, None)
        text = result.get("text", "")
        source_name = result.get("source", {}).get("source_name", "Unknown")

        print(f"[{idx}] Chunk PK: {pk}")
        print(f"    Source: {source_name}")

        if summary:
            chunks_with_summary += 1
            print(f"    ✓ Summary found ({len(summary)} chars)")
            print(f"    Summary preview: {summary[:100]}...")
        else:
            chunks_without_summary += 1
            print(f"    ✗ No summary found")

        print(f"    Text preview: {text[:80]}...")
        print()

    print(f"{'=' * 80}")
    print(f"Results:")
    print(f"  - Chunks with summary: {chunks_with_summary}/{len(results)}")
    print(f"  - Chunks without summary: {chunks_without_summary}/{len(results)}")

    if chunks_with_summary == len(results):
        print(f"\n✓ All sampled chunks have summaries!")
        return 0
    elif chunks_with_summary > 0:
        print(f"\n⚠ Some chunks have summaries, some don't.")
        return 1
    else:
        print(f"\n✗ No chunks have summaries.")
        return 2


if __name__ == "__main__":
    sys.exit(main())
