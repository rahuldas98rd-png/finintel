"""One-shot migration: copy all points from local Qdrant to Qdrant Cloud.

Usage:
    1. Set your Qdrant Cloud credentials in environment:
         export QDRANT_URL=https://xxx.cloud.qdrant.io
         export QDRANT_API_KEY=...
    2. Make sure your local Qdrant Docker container is running on localhost:6333
       and contains the indexed `finintel_chunks` collection.
    3. Run:
         uv run python scripts/migrate_to_qdrant_cloud.py

This script:
    - Connects to local Qdrant at http://localhost:6333 (hardcoded as source)
    - Connects to the cloud cluster via env vars (destination)
    - Recreates the destination collection (drops it if present)
    - Scrolls through every local point (vector + payload) in batches of 100
    - Upserts to the cloud, preserving point IDs so deterministic UUIDv5
      keying is maintained
    - Verifies destination point count matches source
"""
from __future__ import annotations

import os
import sys

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

LOCAL_URL = "http://localhost:6333"
COLLECTION = "finintel_chunks"
BATCH_SIZE = 100


def main() -> None:
    cloud_url = os.environ.get("QDRANT_URL")
    cloud_api_key = os.environ.get("QDRANT_API_KEY")

    if not cloud_url or not cloud_api_key:
        print("ERROR: QDRANT_URL and QDRANT_API_KEY must both be set in the environment.")
        print("       These should point to your Qdrant Cloud cluster, not local.")
        sys.exit(1)

    if "localhost" in cloud_url or "127.0.0.1" in cloud_url:
        print(f"ERROR: QDRANT_URL='{cloud_url}' looks like a local address.")
        print("       Set it to your Qdrant Cloud cluster URL.")
        sys.exit(1)

    print("=" * 70)
    print("Qdrant migration: local → cloud")
    print("=" * 70)
    print(f"Source:      {LOCAL_URL}")
    print(f"Destination: {cloud_url}")
    print(f"Collection:  {COLLECTION}")
    print()

    # Connect to both
    try:
        local = QdrantClient(url=LOCAL_URL)
    except Exception as e:
        print(f"ERROR: cannot connect to local Qdrant at {LOCAL_URL}")
        print(f"       Is your Docker container running? ({e})")
        sys.exit(1)

    try:
        cloud = QdrantClient(url=cloud_url, api_key=cloud_api_key)
    except Exception as e:
        print(f"ERROR: cannot connect to Qdrant Cloud at {cloud_url}")
        print(f"       Check QDRANT_URL and QDRANT_API_KEY. ({e})")
        sys.exit(1)

    # Verify source collection exists
    if not local.collection_exists(COLLECTION):
        print(f"ERROR: collection '{COLLECTION}' not found on local Qdrant.")
        print("       Run your indexing pipeline first.")
        sys.exit(1)

    info = local.get_collection(COLLECTION)
    n_source_points = info.points_count
    vector_dim = info.config.params.vectors.size
    print(f"Source collection has {n_source_points} points, vector_dim={vector_dim}")
    print()

    # Recreate destination collection
    if cloud.collection_exists(COLLECTION):
        print(f"Destination collection '{COLLECTION}' already exists. Deleting...")
        cloud.delete_collection(COLLECTION)

    cloud.create_collection(
        collection_name=COLLECTION,
        vectors_config=VectorParams(size=vector_dim, distance=Distance.COSINE),
    )
    print(f"Created destination collection '{COLLECTION}' (dim={vector_dim}, COSINE)")
    
    # Recreate payload indexes — required for filter conditions in search()
    from qdrant_client.models import PayloadSchemaType
    for field_name in ("ticker", "section"):
        cloud.create_payload_index(
            collection_name=COLLECTION,
            field_name=field_name,
            field_schema=PayloadSchemaType.KEYWORD,
        )
    print(f"Created payload indexes: ticker, section")
    print()

    # Scroll through source and upsert to destination
    print("Migrating points...")
    offset = None
    total_migrated = 0
    while True:
        points, offset = local.scroll(
            collection_name=COLLECTION,
            limit=BATCH_SIZE,
            offset=offset,
            with_vectors=True,
            with_payload=True,
        )
        if not points:
            break

        cloud_points = [
            PointStruct(id=p.id, vector=p.vector, payload=p.payload)
            for p in points
        ]
        cloud.upsert(collection_name=COLLECTION, points=cloud_points)

        total_migrated += len(cloud_points)
        print(f"  Migrated {total_migrated:,} / {n_source_points:,} points...")

        if offset is None:
            break

    print()
    print("Verifying destination...")
    cloud_info = cloud.get_collection(COLLECTION)
    n_dest_points = cloud_info.points_count
    print(f"  Destination has {n_dest_points} points")

    if n_dest_points == n_source_points:
        print()
        print("✓ Migration complete. Counts match.")
        print()
        print("Next steps:")
        print("  1. Update your local .env to point QDRANT_URL/QDRANT_API_KEY at cloud")
        print("  2. Test locally: uv run streamlit run src/finintel/ui/app.py")
        print("  3. Once verified, push to HF Space")
    else:
        print()
        print(f"⚠ MISMATCH: source={n_source_points}, destination={n_dest_points}")
        print("  Re-run the migration; previous attempt may have been interrupted.")
        sys.exit(1)


if __name__ == "__main__":
    main()
