#!/bin/bash
# scripts/git-push-all.sh
# Pushes all branches and tags to all configured remotes.

echo "🔄 Starting global push to all remotes..."

# Get all remotes
REMOTES=$(git remote)

if [ -z "$REMOTES" ]; then
    echo "❌ No remotes found."
    exit 1
fi

for REMOTE in $REMOTES; do
    echo "🚀 Pushing to $REMOTE..."
    # Push all branches
    git push "$REMOTE" --all
    # Push all tags
    git push "$REMOTE" --tags
done

echo "✅ All remotes updated successfully."
