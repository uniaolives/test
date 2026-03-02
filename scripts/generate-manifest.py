#!/usr/bin/env python3
# scripts/generate-manifest.py
import yaml
import os
from pathlib import Path

def generate_manifest(workspace_path):
    """Generate .workspace-manifest.yaml from current structure"""

    workspace = Path(workspace_path)
    manifest = {
        "version": "2.0",
        "strategy": "multi-repo-with-manifest",
        "repositories": {},
        "dependencies": {},
        "health_checks": {
            "nested_repos": 0,
            "ai_commit_coverage": "target: 80%",
            "secret_rotation": "24h"
        }
    }

    # Scan for potential repository directories
    # We'll look for directories that have a Cargo.toml or are significant
    significant_dirs = ["rust", "aigmi", "safecore-9d", "harmonia", "web777_ontology"]

    for dir_name in significant_dirs:
        dir_path = workspace / dir_name
        if dir_path.is_dir():
            manifest["repositories"][dir_name] = {
                "path": dir_name,
                "strategy": "submodule"
            }

    # Write manifest
    with open(workspace / ".workspace-manifest.yaml", "w") as f:
        yaml.dump(manifest, f, default_flow_style=False)

    print("✅ Generated .workspace-manifest.yaml")

if __name__ == "__main__":
    generate_manifest(".")
