#!/bin/bash
set -e

VERSION="31.2-omega"
DIST_NAME="sasc-$VERSION"
BUILD_DIR=$(pwd)/temp_build
ARCHIVE_NAME="sasc-v31.2-omega.tar.gz"

echo "--- Generating SASC Distribution Package ---"

# 1. Clean previous build
rm -rf "$BUILD_DIR"
rm -f "$ARCHIVE_NAME"

# 2. Run install to temp directory
echo "Building and collecting artifacts..."
PREFIX="$BUILD_DIR" ./scripts/install.sh

# 3. Create the archive
echo "Creating archive $ARCHIVE_NAME..."
cd "$BUILD_DIR"
tar -czf "../$ARCHIVE_NAME" .
cd ..

# 4. Cleanup
rm -rf "$BUILD_DIR"

echo "--- Distribution package generated: $ARCHIVE_NAME ---"
