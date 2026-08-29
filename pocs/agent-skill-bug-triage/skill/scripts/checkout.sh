#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:-}"
if [ -z "$TARGET" ]; then
  echo "usage: checkout.sh <repo url | pr url | branch>" >&2
  exit 1
fi

BASE="${TMPDIR:-/tmp}"
BASE="${BASE%/}"

if [ "$TARGET" = "--clean" ]; then
  rm -rf "$BASE"/bug-triage-src-*
  git worktree prune >/dev/null 2>&1 || true
  echo "CLEANED $BASE/bug-triage-src-*"
  exit 0
fi

slug() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | sed 's|[^a-z0-9]|-|g; s|--*|-|g; s|^-||; s|-$||' | cut -c1-48
}

clone_repo() {
  local url="$1"
  local ref="$2"
  local dest="$3"
  rm -rf "$dest"
  if command -v gh >/dev/null 2>&1 && echo "$url" | grep -q "github.com"; then
    gh repo clone "$url" "$dest" -- --depth 50 >/dev/null 2>&1 || git clone --depth 50 "$url" "$dest" >/dev/null 2>&1 || true
  else
    git clone --depth 50 "$url" "$dest" >/dev/null 2>&1 || true
  fi
  if [ ! -d "$dest/.git" ]; then
    echo "ERROR: could not clone $url - check the url and your access to it" >&2
    exit 1
  fi
  if [ -n "$ref" ]; then
    if ! git -C "$dest" fetch --depth 50 origin "$ref" >/dev/null 2>&1; then
      echo "ERROR: could not fetch $ref from $url" >&2
      exit 1
    fi
    git -C "$dest" checkout -q FETCH_HEAD
  fi
}

case "$TARGET" in
  *github.com/*/pull/*|*/pull/*)
    OWNER_REPO=$(echo "$TARGET" | sed -E 's|.*github.com/([^/]+/[^/]+)/pull/.*|\1|')
    NUM=$(echo "$TARGET" | sed -E 's|.*/pull/([0-9]+).*|\1|')
    DEST="$BASE/bug-triage-src-$(slug "$OWNER_REPO-pr-$NUM")"
    clone_repo "https://github.com/$OWNER_REPO.git" "pull/$NUM/head" "$DEST"
    echo "CHECKOUT $DEST"
    echo "REF pull/$NUM/head"
    git -C "$DEST" log --oneline -1
    ;;
  http*|git@*|file://*|ssh://*)
    CLEAN="${TARGET%/}"
    CLEAN="${CLEAN%.git}"
    NAME="$(basename "$(dirname "$CLEAN")")-$(basename "$CLEAN")"
    DEST="$BASE/bug-triage-src-$(slug "$NAME")"
    clone_repo "$TARGET" "" "$DEST"
    echo "CHECKOUT $DEST"
    echo "REF $(git -C "$DEST" rev-parse --abbrev-ref HEAD)"
    git -C "$DEST" log --oneline -1
    ;;
  *)
    if ! git rev-parse --git-dir >/dev/null 2>&1; then
      echo "ERROR: not inside a git repo and '$TARGET' is not a url" >&2
      exit 1
    fi
    ROOT=$(git rev-parse --show-toplevel)
    DEST="$BASE/bug-triage-src-$(slug "$(basename "$ROOT")-$TARGET")"
    git worktree remove --force "$DEST" >/dev/null 2>&1 || true
    rm -rf "$DEST"
    git fetch --all --quiet >/dev/null 2>&1 || true
    git worktree add --detach "$DEST" "$TARGET" >/dev/null 2>&1 || git worktree add --detach "$DEST" "origin/$TARGET" >/dev/null 2>&1 || true
    if [ ! -e "$DEST/.git" ]; then
      echo "ERROR: '$TARGET' is not a branch, tag or commit in $ROOT" >&2
      exit 1
    fi
    echo "CHECKOUT $DEST"
    echo "REF $TARGET"
    git -C "$DEST" log --oneline -1
    ;;
esac
