#!/usr/bin/env bash
# One-shot migration: clone projeto-bioacustico, repush all branches/tags to biotuts.
# Run on your local machine where you're authenticated to GitHub (gh auth login, or
# https with credential helper, or ssh keys).

set -euo pipefail

OLD_URL="https://github.com/Lynnbrosa/projeto-bioacustico.git"
NEW_URL="https://github.com/Lynnbrosa/biotuts.git"
WORK_DIR="${HOME}/.cache/biotuts-migration"

echo "==> Cloning ${OLD_URL} into ${WORK_DIR}"
rm -rf "${WORK_DIR}"
git clone --mirror "${OLD_URL}" "${WORK_DIR}"
cd "${WORK_DIR}"

echo "==> Removing claude/* refs and refs/pull/* (don't migrate)"
git for-each-ref --format='%(refname)' refs/heads/claude/ refs/pull/ | \
    xargs -r -n1 git update-ref -d

echo "==> Pointing remote at ${NEW_URL}"
git remote set-url origin "${NEW_URL}"

echo "==> Pushing all refs (branches + tags)"
git push --mirror

echo
echo "Done. Verify at https://github.com/Lynnbrosa/biotuts"
echo "After confirming biotuts is correct, you can delete projeto-bioacustico at:"
echo "  https://github.com/Lynnbrosa/projeto-bioacustico/settings"
