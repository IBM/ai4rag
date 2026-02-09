#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# Copyright IBM Corp. 2025-2026
# SPDX-License-Identifier: Apache-2.0
# -----------------------------------------------------------------------------
#
# Release Tag Creation Script
#
# Automatically creates and pushes a git tag based on the version
# defined in ai4rag.__version__.
#
# Usage:
#   ./scripts/create_release_tag.sh              # Create tag locally
#   ./scripts/create_release_tag.sh --push       # Create tag and push to remote
#   ./scripts/create_release_tag.sh --dry-run    # Show what would be created
#   ./scripts/create_release_tag.sh --message "Release notes"  # Custom tag message
#

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Default options
PUSH=false
DRY_RUN=false
TAG_MESSAGE=""

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --push)
            PUSH=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --message|-m)
            TAG_MESSAGE="$2"
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --push           Push tag to remote after creation"
            echo "  --message, -m    Custom tag annotation message"
            echo "  --dry-run        Show what would be created without actually creating"
            echo "  -h, --help       Show this help message"
            echo ""
            echo "Examples:"
            echo "  $0                                    # Create tag locally"
            echo "  $0 --push                             # Create and push tag"
            echo "  $0 -m \"Bug fixes and improvements\"    # Create with custom message"
            exit 0
            ;;
        *)
            echo -e "${RED}Error: Unknown option $1${NC}"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Get the script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# Change to project root
cd "$PROJECT_ROOT"

# Read version from ai4rag/__init__.py
VERSION_FILE="$PROJECT_ROOT/ai4rag/__init__.py"

if [[ ! -f "$VERSION_FILE" ]]; then
    echo -e "${RED}Error: Cannot find $VERSION_FILE${NC}"
    exit 1
fi

# Extract version using grep and sed
VERSION=$(grep '^__version__' "$VERSION_FILE" | sed 's/__version__[[:space:]]*=[[:space:]]*"\(.*\)"/\1/')

if [[ -z "$VERSION" ]]; then
    echo -e "${RED}Error: Could not extract version from $VERSION_FILE${NC}"
    exit 1
fi

# Create tag name
TAG_NAME="v$VERSION"

echo -e "${BLUE}Creating release tag for version: $VERSION${NC}"

# Validate semantic versioning (X.Y.Z)
if ! [[ "$VERSION" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]]; then
    echo -e "${YELLOW}Warning: Version '$VERSION' doesn't follow semantic versioning (X.Y.Z)${NC}"
    read -p "Continue anyway? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Tag creation cancelled."
        exit 1
    fi
fi

# Check if tag already exists
if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
    echo -e "${RED}Error: Tag $TAG_NAME already exists${NC}"
    echo ""
    echo "To delete the existing tag:"
    echo "  git tag -d $TAG_NAME                    # Delete locally"
    echo "  git push origin :refs/tags/$TAG_NAME    # Delete from remote"
    exit 1
fi

# Get current branch
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD)

# Check for uncommitted changes
if [[ -n $(git status -s) ]]; then
    echo -e "${YELLOW}Warning: You have uncommitted changes${NC}"
    git status -s
    echo ""
    read -p "Continue with tag creation? [y/N]: " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        echo "Tag creation cancelled."
        exit 1
    fi
fi

# Set default tag message if not provided
if [[ -z "$TAG_MESSAGE" ]]; then
    TAG_MESSAGE="Release $TAG_NAME"
fi

# Show tag creation plan
echo ""
echo "============================================================"
echo "TAG CREATION PLAN"
echo "============================================================"
echo "Tag Name:       $TAG_NAME"
echo "Current Branch: $CURRENT_BRANCH"
echo "Commit:         $(git rev-parse --short HEAD)"
echo "Message:        $TAG_MESSAGE"
echo "Push to Remote: $([[ "$PUSH" == true ]] && echo "Yes" || echo "No")"
echo "============================================================"
echo ""

if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}DRY RUN - Commands that would be executed:${NC}"
    echo "  git tag -a $TAG_NAME -m \"$TAG_MESSAGE\""
    if [[ "$PUSH" == true ]]; then
        echo "  git push origin $TAG_NAME"
    fi
    echo ""
    echo "No changes made."
    exit 0
fi

# Create annotated tag
echo "Creating annotated tag..."
if git tag -a "$TAG_NAME" -m "$TAG_MESSAGE"; then
    echo -e "${GREEN}✓ Tag $TAG_NAME created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create tag${NC}"
    exit 1
fi

# Push tag if requested
if [[ "$PUSH" == true ]]; then
    echo ""
    echo "Pushing tag to remote..."
    if git push origin "$TAG_NAME"; then
        echo -e "${GREEN}✓ Tag $TAG_NAME pushed to remote${NC}"
    else
        echo -e "${RED}✗ Failed to push tag${NC}"
        echo ""
        echo "Tag was created locally. To push manually:"
        echo "  git push origin $TAG_NAME"
        exit 1
    fi
fi

# Show success summary
echo ""
echo "============================================================"
echo -e "${GREEN}✓ Release tag created successfully!${NC}"
echo "============================================================"

if [[ "$PUSH" == false ]]; then
    echo ""
    echo "To push the tag to remote:"
    echo "  git push origin $TAG_NAME"
    echo ""
    echo "Or re-run with --push flag:"
    echo "  ./scripts/create_release_tag.sh --push"
fi

echo ""
echo "Available tags:"
git tag -l -n1

echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "  1. The tag will trigger GitHub Actions to deploy documentation"
echo "  2. Create a GitHub release (manually or via gh CLI)"
echo "  3. Publish to PyPI if configured"
