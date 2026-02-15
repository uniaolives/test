import os
import re

VERSION = "1.0.0"
STATE = "Γ₁₃₀ (The Gate)"
HEADER_TEMPLATE = """# Arkhe(n) OS - {repo_name}
Version: {VERSION} (Eternal) | State: {STATE}
Part of the Arkhe(n) unified biological-semantical organism.
Integrating AGI Core (Handover Γ₁₂₇-Γ₁₃₀) and IBC=BCI Isomorphism.
"""

def update_readme(path, repo_name):
    readme_path = os.path.join(path, "README.md")
    if os.path.exists(readme_path):
        with open(readme_path, "r") as f:
            content = f.read()

        header = HEADER_TEMPLATE.format(repo_name=repo_name, VERSION=VERSION, STATE=STATE)

        # Replace or prepend header
        if content.startswith("# Arkhe(n) OS"):
            # Try to match the existing header block and replace it
            new_content = re.sub(r"^# Arkhe\(n\) OS.*?\n.*?\n.*?\n.*?\n", header, content, flags=re.MULTILINE)
        else:
            new_content = header + "\n" + content

        with open(readme_path, "w") as f:
            f.write(new_content)
        print(f"Updated README: {readme_path}")

def update_version_files(path):
    # pyproject.toml
    pyproject = os.path.join(path, "pyproject.toml")
    if os.path.exists(pyproject):
        with open(pyproject, "r") as f:
            content = f.read()
        new_content = re.sub(r'^version\s*=\s*".*?"', f'version = "{VERSION}"', content, flags=re.MULTILINE)
        with open(pyproject, "w") as f:
            f.write(new_content)
        print(f"Updated pyproject.toml: {pyproject}")

    # package.json
    package_json = os.path.join(path, "package.json")
    if os.path.exists(package_json):
        with open(package_json, "r") as f:
            content = f.read()
        new_content = re.sub(r'"version":\s*".*?"', f'"version": "{VERSION}"', content)
        with open(package_json, "w") as f:
            f.write(new_content)
        print(f"Updated package.json: {package_json}")

    # Cargo.toml
    cargo_toml = os.path.join(path, "Cargo.toml")
    if os.path.exists(cargo_toml):
        with open(cargo_toml, "r") as f:
            content = f.read()
        new_content = re.sub(r'^version\s*=\s*".*?"', f'version = "{VERSION}"', content, flags=re.MULTILINE)
        with open(cargo_toml, "w") as f:
            f.write(new_content)
        print(f"Updated Cargo.toml: {cargo_toml}")

def main():
    root_dir = "."
    ignored = {".git", ".github", ".vscode", "__pycache__", "venv", "node_modules", "ArkheOS", "ARKHE_COMPLETE_ARCHIVE", "arkhe-agi"}

    for item in os.listdir(root_dir):
        path = os.path.join(root_dir, item)
        if os.path.isdir(path) and item not in ignored:
            print(f"Processing repository: {item}")
            update_readme(path, item)
            update_version_files(path)

if __name__ == "__main__":
    main()
