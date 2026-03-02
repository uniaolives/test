#!/bin/bash
# complete-migration.sh
# One script to migrate the OpenClaw workspace to the new hybrid architecture

echo "🚀 OpenClaw Workspace Migration Suite"
echo "===================================="

# Step 1: Emergency cleanup
echo "📦 Step 1: Emergency nesting cleanup..."
./scripts/emergency-cleanup.sh

# Step 2: Generate manifest
echo "📋 Step 2: Generating workspace manifest..."
python3 scripts/generate-manifest.py

# Step 3: Install AI hooks
echo "🤖 Step 3: Installing AI commit hooks..."
./scripts/install-ai-hooks.sh

# Step 4: Migrate secrets
echo "🔒 Step 4: Migrating secrets to quantum vault..."
python3 scripts/migrate-secrets.py

# Step 5: Validate structure
echo "✅ Step 5: Validating new structure..."
python3 scripts/validate-structure.py

# Step 6: Generate dashboard
echo "📊 Step 6: Generating health dashboard..."
python3 scripts/workspace-dashboard.py

echo ""
echo "🎉 Migration complete!"
echo ""
echo "📋 Post-migration checklist:"
echo "   1. Review .workspace-manifest.yaml for accuracy"
echo "   2. Test git operations in each project"
echo "   3. Verify AI hooks are working with test commit"
echo "   4. Backup quantum vault keys to secure location"
echo "   5. Schedule weekly health checks"
