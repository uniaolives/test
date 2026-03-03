#!/usr/bin/env python3
# scripts/migrate-dependency.py
import subprocess
import sys
from pathlib import Path
import yaml

class DependencyMigrator:
    def __init__(self, component_path):
        self.path = Path(component_path)
        self.workspace_root = Path(".")
        self.analysis = self._analyze_component()

    def _analyze_component(self):
        """Analyze change frequency and dependencies"""
        try:
            # Get commit frequency (last 30 days)
            cmd = f"git -C {self.path} log --since='30 days ago' --oneline 2>/dev/null | wc -l"
            commits = int(subprocess.check_output(cmd, shell=True).strip())
        except:
            commits = 0

        # Get dependent projects from manifest
        deps = []
        manifest_path = self.workspace_root / ".workspace-manifest.yaml"
        if manifest_path.exists():
            with open(manifest_path) as f:
                manifest = yaml.safe_load(f)

            for project, config in manifest.get("repositories", {}).items():
                if self.path.name in config.get("dependencies", []):
                    deps.append(project)

        return {
            "change_frequency": commits / 30,  # commits/day
            "dependent_projects": len(deps),
            "is_infrastructure": self.path.name in ["configs", "schemas", "security"]
        }

    def recommend_strategy(self):
        analysis = self.analysis

        if analysis["change_frequency"] > 5 or analysis["dependent_projects"] > 3:
            return "subtree", "High change rate with multiple dependents"
        elif analysis["is_infrastructure"] and analysis["change_frequency"] < 1:
            return "submodule", "Stable infrastructure component"
        else:
            return "package", "Independent component suitable for packaging"

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: ./scripts/migrate-dependency.py <component_path>")
        sys.exit(1)

    migrator = DependencyMigrator(sys.argv[1])
    strategy, reason = migrator.recommend_strategy()
    print(f"Recommended strategy for {sys.argv[1]}: {strategy}")
    print(f"Reason: {reason}")
