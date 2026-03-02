#!/bin/bash
# emergency-cleanup.sh
# Fixes nested Git repositories by converting them to submodules or moving to shared/

WORKSPACE_ROOT=$(pwd)
echo "🔍 Scanning for nested Git repositories in $WORKSPACE_ROOT..."

# Find all .git directories that are NOT the root one
NESTED_REPOS=$(find . -mindepth 2 -name ".git" -type d)

if [ -z "$NESTED_REPOS" ]; then
    echo "✅ No nested repos found. System is clean."
    exit 0
fi

echo "⚠️  Found nested repositories:"
echo "$NESTED_REPOS"

# Create backup
BACKUP_DIR="/tmp/openclaw-backup-$(date +%s)"
echo "📦 Creating backup at $BACKUP_DIR"
mkdir -p "$BACKUP_DIR"
cp -r . "$BACKUP_DIR"

# Fix each nested repo
for repo in $NESTED_REPOS; do
    repo_path=$(dirname "$repo")
    repo_name=$(basename "$repo_path")

    echo "🔄 Processing $repo_name..."

    # Check if remote exists before removing .git
    REMOTE_URL=$(git -C "$repo_path" remote get-url origin 2>/dev/null)

    rm -rf "$repo"

    if [ ! -z "$REMOTE_URL" ]; then
        echo "   Converting to submodule: $REMOTE_URL"
        git submodule add "$REMOTE_URL" "$repo_path"
    else
        echo "   No remote found. Leaving as plain directory (tracked by parent)."
    fi
done

echo "✅ Emergency cleanup complete!"
