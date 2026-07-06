set -eu
printf '%s\n' 'Uninstall Bug Cataloger from:' '1) Claude Code' '2) Codex' '3) Both'
printf 'Choice: '
read -r choice
case "$choice" in
  1) targets=("$HOME/.claude/skills") ;;
  2) targets=("${CODEX_HOME:-$HOME/.codex}/skills") ;;
  3) targets=("$HOME/.claude/skills" "${CODEX_HOME:-$HOME/.codex}/skills") ;;
  *) printf '%s\n' 'Invalid choice' >&2; exit 1 ;;
esac
for target in "${targets[@]}"; do
  rm -rf "$target/bug-cataloger"
  printf 'Removed: %s\n' "$target/bug-cataloger"
done
