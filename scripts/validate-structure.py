#!/usr/bin/env python3
# scripts/validate-structure.py
import os
import sys
from pathlib import Path

class WorkspaceValidator:
    def __init__(self, workspace_path):
        self.workspace = Path(workspace_path).resolve()

    def validate(self):
        issues = []

        # Check 1: No nested .git directories (excluding submodules which are often just files or different paths)
        # In a clean manifest-based workspace, we only want the root .git
        for root, dirs, files in os.walk(self.workspace):
            if '.git' in dirs and Path(root) != self.workspace:
                # If it's a directory, it's a nested repo (bad)
                # If it's a file, it's likely a submodule (okay)
                git_path = Path(root) / '.git'
                if git_path.is_dir():
                    issues.append(f"Nested Git directory found at: {root}")

        # Check 2: Manifest exists
        manifest = self.workspace / ".workspace-manifest.yaml"
        if not manifest.exists():
            issues.append("Missing .workspace-manifest.yaml")

        return issues

if __name__ == "__main__":
    validator = WorkspaceValidator(".")
    issues = validator.validate()

    if issues:
        print("❌ Validation failed:")
        for issue in issues:
            print(f"   - {issue}")
        sys.exit(1)
    else:
        print("✅ Workspace structure is valid")
