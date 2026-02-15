import os

VERSION = "1.0.0"
STATE = "Γ₁₃₀ (The Gate)"
AGI_CONTEXT = "Integrating AGI Core (Handover Γ₁₂₇-Γ₁₃₀) and IBC=BCI Isomorphism."

def update_readme(path, repo_name):
    readme_path = os.path.join(path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            lines = f.readlines()

        if not lines:
            return

        # Ensure the first line is the correct title
        lines[0] = f"# Arkhe(n) OS - {repo_name}\n"

        # Ensure the second line is version and state
        # We'll use Γ₁₃₀ as the baseline for this upgrade
        lines[1] = f"Version: {VERSION} (Eternal) | State: {STATE}\n"

        # Ensure the third and fourth lines are the standard Arkhe lines
        standard_line3 = "Part of the Arkhe(n) unified biological-semantical organism.\n"
        standard_line4 = AGI_CONTEXT + "\n"

        if len(lines) < 3:
            lines.append(standard_line3)
        else:
            lines[2] = standard_line3

        if len(lines) < 4:
            lines.append(standard_line4)
        else:
            # Check if line 4 is already the AGI context, if not, insert it or replace it if it's an old header line
            if AGI_CONTEXT not in lines[3]:
                lines[3] = standard_line4

        with open(readme_path, "w") as f:
            f.writelines(lines)
        print(f"Verified README: {readme_path}")

def main():
    root_dir = "."
    ignored = {".git", ".github", ".vscode", "__pycache__", "venv", "node_modules", "ArkheOS", "ARKHE_COMPLETE_ARCHIVE", "arkhe-agi"}

    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path) and item not in ignored:
            update_readme(path, item)

if __name__ == "__main__":
    main()
