"""One-shot migration: backfill started_at on existing catalog entries.

Reads created_at from run_metadata.json5 for each entry.
Safe to re-run — skips entries that already have started_at.
"""

import json
import sys
from pathlib import Path

try:
    import pyjson5
except ImportError:
    print("pyjson5 required: pip install pyjson5")
    sys.exit(1)

CATALOG_PATH = Path("logs/catalog_data/catalog.json")


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}")
        sys.exit(1)

    with open(CATALOG_PATH) as f:
        data = json.load(f)

    updated = 0
    skipped = 0
    missing = 0

    for entry in data["entries"]:
        if entry.get("started_at"):
            skipped += 1
            continue

        run_dir = Path(entry["run_dir"])
        meta_file = run_dir / "run_metadata.json5"
        if not meta_file.exists():
            # Try plain json
            meta_file = run_dir / "run_metadata.json"

        if meta_file.exists():
            try:
                with open(meta_file) as f:
                    if meta_file.suffix == ".json5":
                        meta = pyjson5.load(f)
                    else:
                        meta = json.load(f)
                created_at = meta.get("created_at")
                if created_at:
                    entry["started_at"] = created_at
                    updated += 1
                    print(f"  {entry['run_id']}: {created_at[:16]}")
                else:
                    entry["started_at"] = None
                    missing += 1
                    print(f"  {entry['run_id']}: no created_at in metadata")
            except Exception as e:
                entry["started_at"] = None
                missing += 1
                print(f"  {entry['run_id']}: error reading metadata: {e}")
        else:
            entry["started_at"] = None
            missing += 1
            print(f"  {entry['run_id']}: no metadata file at {run_dir}")

    with open(CATALOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Updated: {updated}, Skipped (already set): {skipped}, Missing: {missing}")
    print(f"Catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
