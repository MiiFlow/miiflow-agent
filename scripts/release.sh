#!/bin/bash
# Release script for miiflow-llm
# Usage: ./scripts/release.sh [patch|minor|major|version]
#
# Examples:
#   ./scripts/release.sh patch      # 0.1.0 → 0.1.1
#   ./scripts/release.sh minor      # 0.1.0 → 0.2.0
#   ./scripts/release.sh major      # 0.1.0 → 1.0.0
#   ./scripts/release.sh 0.2.0      # Set specific version

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Check we're in the right directory
if [ ! -f "pyproject.toml" ]; then
    echo -e "${RED}Error: Must run from packages/miiflow-llm directory${NC}"
    exit 1
fi

# Get version bump type
BUMP_TYPE=${1:-patch}

echo -e "${YELLOW}=== miiflow-llm Release Script ===${NC}"
echo ""

# Get current version
CURRENT_VERSION=$(poetry version -s)
echo "Current version: $CURRENT_VERSION"

# Bump version
if [[ "$BUMP_TYPE" =~ ^[0-9]+\.[0-9]+\.[0-9]+ ]]; then
    poetry version "$BUMP_TYPE"
else
    poetry version "$BUMP_TYPE"
fi

NEW_VERSION=$(poetry version -s)
echo -e "New version: ${GREEN}$NEW_VERSION${NC}"
echo ""

# Confirm
read -p "Continue with release v$NEW_VERSION? (y/n) " -n 1 -r
echo ""
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "Aborting. Reverting version..."
    poetry version "$CURRENT_VERSION"
    exit 1
fi

# Run tests
echo ""
echo -e "${YELLOW}Running tests...${NC}"
poetry run pytest tests/ -v
echo -e "${GREEN}Tests passed!${NC}"

# Build
echo ""
echo -e "${YELLOW}Building package...${NC}"
poetry build
echo -e "${GREEN}Build complete!${NC}"

# Git operations
echo ""
echo -e "${YELLOW}Committing version bump...${NC}"
git add pyproject.toml
git commit -m "chore: bump miiflow-llm version to $NEW_VERSION"

echo -e "${YELLOW}Creating git tag...${NC}"
git tag "miiflow-llm-v$NEW_VERSION"

# Ask about publishing
echo ""
read -p "Push to origin and publish to PyPI? (y/n) " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo -e "${YELLOW}Pushing to origin...${NC}"
    git push origin main
    git push origin "miiflow-llm-v$NEW_VERSION"

    echo -e "${YELLOW}Publishing to PyPI...${NC}"
    poetry publish

    echo ""
    echo -e "${GREEN}=== Release v$NEW_VERSION complete! ===${NC}"
    echo "PyPI: https://pypi.org/project/miiflow-llm/$NEW_VERSION/"
else
    echo ""
    echo -e "${YELLOW}Skipped push/publish. To complete manually:${NC}"
    echo "  git push origin main"
    echo "  git push origin miiflow-llm-v$NEW_VERSION"
    echo "  poetry publish"
fi
