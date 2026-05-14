#!/usr/bin/env bash
set -euo pipefail

SRC_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$HOME/.mcp/repo-tool"
BIN="$ROOT/bin/repo-mcp"
CLAUDE_CONFIG="$HOME/.claude.json"

NODE_MAJOR="$(node -p 'process.versions.node.split(".")[0]' 2>/dev/null || echo 0)"
if [ "$NODE_MAJOR" -lt 20 ]; then
  echo "Need Node >= 20 (have ${NODE_MAJOR})"
  exit 1
fi
command -v git >/dev/null || { echo "Need git on PATH"; exit 1; }
command -v rg  >/dev/null || { echo "Need ripgrep on PATH (brew install ripgrep | apt install ripgrep)"; exit 1; }

mkdir -p "$ROOT/bin" "$ROOT/repos"

if [ ! -f "$ROOT/registry.json" ]; then
  printf '{\n  "version": 1,\n  "repos": []\n}\n' > "$ROOT/registry.json"
fi

rm -rf "$ROOT/src" "$ROOT/dist"
cp -R "$SRC_DIR/src" "$ROOT/"
cp "$SRC_DIR/package.json" "$ROOT/"
cp "$SRC_DIR/tsconfig.json" "$ROOT/"

cd "$ROOT"
npm install --silent
npm run build --silent

cat > "$BIN" <<EOF
#!/usr/bin/env node
import('$ROOT/dist/index.js');
EOF
chmod +x "$BIN"

node -e "
const fs = require('fs');
const p = '$CLAUDE_CONFIG';
let cfg = {};
if (fs.existsSync(p)) {
  try { cfg = JSON.parse(fs.readFileSync(p, 'utf8')); } catch (e) { cfg = {}; }
}
cfg.mcpServers = cfg.mcpServers || {};
if (!cfg.mcpServers['repo-mcp']) {
  cfg.mcpServers['repo-mcp'] = { command: '$BIN', args: [] };
  fs.writeFileSync(p, JSON.stringify(cfg, null, 2));
  console.log('Registered repo-mcp in ' + p);
} else {
  console.log('repo-mcp already registered in ' + p);
}
"

ADDED=0
while true; do
  if [ "$ADDED" -eq 0 ]; then
    printf 'Add a GitHub repo now? [y/N] '
  else
    printf 'Add another GitHub repo? [y/N] '
  fi
  read -r yn
  case "$yn" in
    [Yy]*)
      printf 'GitHub URL: '
      read -r URL
      printf 'Branch (optional, blank = default): '
      read -r BRANCH
      if [ -z "$BRANCH" ]; then
        node "$ROOT/dist/cli-add.js" "$URL" && ADDED=$((ADDED+1)) || echo "add_repo failed"
      else
        node "$ROOT/dist/cli-add.js" "$URL" "$BRANCH" && ADDED=$((ADDED+1)) || echo "add_repo failed"
      fi
      ;;
    *) break ;;
  esac
done

echo ""
echo "Installed to $ROOT"
echo "Entry:      $BIN"
echo "Registry:   $ROOT/registry.json"
echo "Re-run install.sh to add more repos, or call add_repo from Claude Code."
echo "Uninstall:  $SRC_DIR/uninstall.sh"
