"""One-shot migration: add tag categories to catalog.json.

Adds `category` field to each CatalogTag and creates TagCategory entries.
Safe to re-run — skips tags that already have a category set.
"""

import json
import sys
from pathlib import Path

CATALOG_PATH = Path("logs/catalog_data/catalog.json")

# Mapping: category_name -> list of tag names
CATEGORY_ASSIGNMENTS: dict[str, list[str]] = {
    "Project": [
        "UnmonitorabilityElicitations",
        "TinkerHyperparameterExplorations",
        "TeacherSpecializationViaKL",
        "ShayanDistillationToAmplifyBias",
    ],
    "Outcome": [
        "AlgorithmicFailure",
        "SystemFailuresCrash",
        "ConfigureationFailure",
        "WorkedSurprisinglyWell",
        "BugDetected",
    ],
    "RunType": [
        "Baseline",
    ],
}

CATEGORY_DESCRIPTIONS: dict[str, str] = {
    "Project": "Which project/research direction this run belongs to",
    "Outcome": "What happened — success, failure mode, or incident",
    "RunType": "The role of this run (baseline, ablation, etc.)",
}


def main() -> None:
    if not CATALOG_PATH.exists():
        print(f"Catalog not found at {CATALOG_PATH}")
        sys.exit(1)

    with open(CATALOG_PATH) as f:
        data = json.load(f)

    # Build reverse map: tag_name -> category
    tag_to_category: dict[str, str] = {}
    for cat_name, tag_names in CATEGORY_ASSIGNMENTS.items():
        for tag_name in tag_names:
            tag_to_category[tag_name] = cat_name

    # Update tags
    updated_count = 0
    for tag in data["tags"]:
        if tag.get("category") and tag["category"] != "Uncategorized":
            continue  # already categorized
        cat = tag_to_category.get(tag["name"], "Uncategorized")
        tag["category"] = cat
        updated_count += 1
        print(f"  {tag['name']} -> {cat}")

    # Add tag_categories
    categories = [
        {"name": name, "description": CATEGORY_DESCRIPTIONS[name]}
        for name in CATEGORY_ASSIGNMENTS
    ]
    data["tag_categories"] = categories

    with open(CATALOG_PATH, "w") as f:
        json.dump(data, f, indent=2)

    print(f"\nDone. Updated {updated_count} tags, wrote {len(categories)} categories.")
    print(f"Catalog: {CATALOG_PATH}")


if __name__ == "__main__":
    main()
