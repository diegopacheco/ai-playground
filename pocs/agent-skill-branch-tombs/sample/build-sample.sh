#!/usr/bin/env bash
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$HERE/graveyard-repo"

rm -rf "$REPO"
mkdir -p "$REPO"
cd "$REPO"

git init -q -b main
git config user.name "Repo Bot"
git config user.email "bot@local"

commit_as () {
  local name="$1" email="$2" date="$3" msg="$4" file="$5"
  echo "$msg" >> "$file"
  git add -A
  GIT_AUTHOR_NAME="$name" GIT_AUTHOR_EMAIL="$email" \
  GIT_COMMITTER_NAME="$name" GIT_COMMITTER_EMAIL="$email" \
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
  git commit -q -m "$msg"
}

dead_branch () {
  local branch="$1" name="$2" email="$3" date="$4" msg="$5"
  git checkout -q -b "$branch" main
  commit_as "$name" "$email" "$date" "$msg" "work.txt"
  git checkout -q main
}

merged_branch () {
  local branch="$1" name="$2" email="$3" date="$4" msg="$5"
  git checkout -q -b "$branch" main
  commit_as "$name" "$email" "$date" "$msg" "work.txt"
  GIT_AUTHOR_NAME="Repo Bot" GIT_AUTHOR_EMAIL="bot@local" \
  GIT_COMMITTER_NAME="Repo Bot" GIT_COMMITTER_EMAIL="bot@local" \
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
  git checkout -q main
  GIT_AUTHOR_NAME="Repo Bot" GIT_AUTHOR_EMAIL="bot@local" \
  GIT_COMMITTER_NAME="Repo Bot" GIT_COMMITTER_EMAIL="bot@local" \
  GIT_AUTHOR_DATE="$date" GIT_COMMITTER_DATE="$date" \
  git merge -q --no-ff -m "merge $branch" "$branch"
}

commit_as "Repo Bot" "bot@local" "2024-01-02T09:00:00" "init project" "README.md"

dead_branch   "feature/login-redesign"  "Alice Chen"   "alice@team.dev"   "2026-06-04T11:00:00" "redesign login screen"
dead_branch   "feature/payments-v2"     "Bob Martins"  "bob@team.dev"     "2026-04-25T14:00:00" "wire new payments api"
merged_branch "chore/deps-bump"         "Dave Okoro"   "dave@team.dev"    "2026-04-10T10:00:00" "bump dependencies"
merged_branch "release/1.4"             "Dave Okoro"   "dave@team.dev"    "2026-03-06T16:00:00" "cut release 1.4"
merged_branch "bugfix/npe-checkout"     "Alice Chen"   "alice@team.dev"   "2026-02-09T09:30:00" "fix npe at checkout"
dead_branch   "experiment/graphql"      "Carol Diaz"   "carol@team.dev"   "2025-11-21T13:00:00" "spike graphql gateway"
dead_branch   "feature/old-dashboard"   "Bob Martins"  "bob@team.dev"     "2025-05-05T15:00:00" "legacy dashboard rework"
dead_branch   "spike/ml-ranking"        "Carol Diaz"   "carol@team.dev"   "2024-05-20T12:00:00" "ml ranking proof"

commit_as "Repo Bot" "bot@local" "2026-06-08T18:00:00" "release notes" "CHANGELOG.md"

git checkout -q main
echo "built sample repo at $REPO"
git --no-pager branch
