#!/usr/bin/env bash

FOUND=0
BASE_DIR="${1:-.}"

check_lockfile() {
  local lockfile="$1"
  local dir
  dir="$(dirname "$lockfile")"

  local malicious_version
  malicious_version=$(grep -E '"version": "1\.14\.1"|"version": "0\.30\.4"' "$lockfile" 2>/dev/null)
  if [ -n "$malicious_version" ]; then
    echo "[ALERT] Malicious axios version found in: $lockfile"
    echo "        $malicious_version"
    FOUND=1
  fi

  local phantom_dep
  phantom_dep=$(grep -E '"plain-crypto-js"' "$lockfile" 2>/dev/null)
  if [ -n "$phantom_dep" ]; then
    echo "[ALERT] Phantom dependency plain-crypto-js found in: $lockfile"
    FOUND=1
  fi

  if [ -d "$dir/node_modules/plain-crypto-js" ]; then
    echo "[ALERT] plain-crypto-js installed in: $dir/node_modules/plain-crypto-js"
    FOUND=1
  fi

  local malicious_axios_pkg="$dir/node_modules/axios/package.json"
  if [ -f "$malicious_axios_pkg" ]; then
    local ver
    ver=$(grep -oE '"version": "(1\.14\.1|0\.30\.4)"' "$malicious_axios_pkg" 2>/dev/null)
    if [ -n "$ver" ]; then
      echo "[ALERT] Compromised axios binary in: $malicious_axios_pkg ($ver)"
      FOUND=1
    fi
  fi
}

check_yarn_lock() {
  local lockfile="$1"
  local dir
  dir="$(dirname "$lockfile")"

  local malicious_version
  malicious_version=$(grep -E 'axios@.*1\.14\.1|axios@.*0\.30\.4|plain-crypto-js' "$lockfile" 2>/dev/null)
  if [ -n "$malicious_version" ]; then
    echo "[ALERT] Malicious entry found in: $lockfile"
    echo "        $malicious_version"
    FOUND=1
  fi

  if [ -d "$dir/node_modules/plain-crypto-js" ]; then
    echo "[ALERT] plain-crypto-js installed in: $dir/node_modules/plain-crypto-js"
    FOUND=1
  fi
}

check_rat_artifacts() {
  local os_type
  os_type="$(uname -s)"
  echo "--- Checking RAT artifacts on $os_type ---"

  if [ "$os_type" = "Darwin" ]; then
    if [ -e "/Library/Caches/com.apple.act.mond" ]; then
      echo "[CRITICAL] RAT artifact found: /Library/Caches/com.apple.act.mond"
      echo "[CRITICAL] Machine is likely compromised. Isolate immediately."
      FOUND=1
    else
      echo "[OK] No macOS RAT artifact found."
    fi
  elif [ "$os_type" = "Linux" ]; then
    if [ -e "/tmp/ld.py" ]; then
      echo "[CRITICAL] RAT artifact found: /tmp/ld.py"
      echo "[CRITICAL] Machine is likely compromised. Isolate immediately."
      FOUND=1
    else
      echo "[OK] No Linux RAT artifact found."
    fi
  else
    echo "[INFO] Unsupported OS for RAT artifact check. Manually check:"
    echo "       Windows: %PROGRAMDATA%\\wt.exe"
  fi
}

echo "=== Axios RAT Detector (CVE-2026-03-31) ==="
echo "Scanning: $(cd "$BASE_DIR" && pwd)"
echo ""

echo "--- Scanning package-lock.json files ---"
LOCK_COUNT=0
while IFS= read -r -d '' lockfile; do
  LOCK_COUNT=$((LOCK_COUNT + 1))
  check_lockfile "$lockfile"
done < <(find "$BASE_DIR" -name "package-lock.json" -not -path "*/node_modules/*" -print0 2>/dev/null)
echo "Scanned $LOCK_COUNT package-lock.json files."
echo ""

echo "--- Scanning yarn.lock files ---"
YARN_COUNT=0
while IFS= read -r -d '' lockfile; do
  YARN_COUNT=$((YARN_COUNT + 1))
  check_yarn_lock "$lockfile"
done < <(find "$BASE_DIR" -name "yarn.lock" -not -path "*/node_modules/*" -print0 2>/dev/null)
echo "Scanned $YARN_COUNT yarn.lock files."
echo ""

check_rat_artifacts
echo ""

if [ "$FOUND" -eq 0 ]; then
  echo "=== RESULT: CLEAN - No indicators of compromise found. ==="
else
  echo "=== RESULT: COMPROMISED - Indicators found! ==="
  echo ""
  echo "Recommended actions:"
  echo "  1. Disconnect machine from network"
  echo "  2. Delete node_modules and lockfiles in affected dirs"
  echo "  3. Pin axios to 1.14.0 (or 0.30.3 for legacy)"
  echo "  4. Rotate ALL credentials (npm tokens, SSH keys, cloud keys, API keys, .env)"
  echo "  5. Run npm install fresh"
  echo "  6. Audit system for further compromise"
fi

exit $FOUND
