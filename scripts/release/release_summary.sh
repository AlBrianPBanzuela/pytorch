#!/bin/bash
# Generate a release summary line like:
#   "This release is composed of 4160 commits from 536 contributors since PyTorch 2.9."
#
# Usage:
#   ./scripts/release/release_summary.sh                    # auto-detect latest release
#   ./scripts/release/release_summary.sh v2.10.0 v2.9.0     # explicit current and previous tags

set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ $# -ge 2 ]; then
    CURRENT_TAG="$1"
    PREVIOUS_TAG="$2"
elif [ $# -eq 1 ]; then
    CURRENT_TAG="$1"
    # Derive previous minor version tag
    version="${CURRENT_TAG#v}"
    major="${version%%.*}"
    rest="${version#*.}"
    minor="${rest%%.*}"
    prev_minor=$((minor - 1))
    PREVIOUS_TAG="v${major}.${prev_minor}.0"
else
    # Auto-detect: find the latest vX.Y.0 release tag
    CURRENT_TAG=$(git tag -l 'v[0-9]*.[0-9]*.0' | sort -V | tail -1)
    if [ -z "$CURRENT_TAG" ]; then
        echo "Error: no release tags found" >&2
        exit 1
    fi
    version="${CURRENT_TAG#v}"
    major="${version%%.*}"
    rest="${version#*.}"
    minor="${rest%%.*}"
    prev_minor=$((minor - 1))
    PREVIOUS_TAG="v${major}.${prev_minor}.0"
fi

# Validate tags exist
for tag in "$CURRENT_TAG" "$PREVIOUS_TAG"; do
    if ! git rev-parse "$tag" >/dev/null 2>&1; then
        echo "Error: tag '$tag' not found" >&2
        exit 1
    fi
done

commits=$(git log --oneline "${PREVIOUS_TAG}..${CURRENT_TAG}" | wc -l | tr -d ' ')
contributors=$(git shortlog -sn "${PREVIOUS_TAG}..${CURRENT_TAG}" | wc -l | tr -d ' ')

# Extract version number for display (e.g., v2.9.0 -> 2.9)
prev_display="${PREVIOUS_TAG#v}"
prev_display="${prev_display%.0}"

echo "This release is composed of ${commits} commits from ${contributors} contributors since PyTorch ${prev_display}."
