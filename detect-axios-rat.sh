#!/usr/bin/env bash

FOUND=0
BASE_DIR="${1:-.}"

check_lockfile() {
  local lockfile="$1"
  local dir
  dir="$(dirname "$lockfile")"

  local malicious_version
  malicious_version=$(node -e "
    const fs = require('fs');
    const lock = JSON.parse(fs.readFileSync('$lockfile', 'utf8'));
    const pkgs = lock.packages || {};
    const deps = lock.dependencies || {};
    for (const [key, val] of Object.entries(pkgs)) {
      if ((key.endsWith('/axios') || key === 'axios') && (val.version === '1.14.1' || val.version === '0.30.4')) {
        console.log(key + ' -> ' + val.version);
      }
    }
    for (const [key, val] of Object.entries(deps)) {
      if (key === 'axios' && (val.version === '1.14.1' || val.version === '0.30.4')) {
        console.log(key + ' -> ' + val.version);
      }
    }
  " 2>/dev/null)
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

check_disk_artifacts() {
  echo "--- Checking macOS disk artifacts ---"
  local clean=1

  if [ -e "/Library/Caches/com.apple.act.mond" ]; then
    echo "[CRITICAL] RAT artifact found: /Library/Caches/com.apple.act.mond"
    FOUND=1
    clean=0
  fi

  if [ -e "$HOME/Library/Caches/com.apple.act.mond" ]; then
    echo "[CRITICAL] RAT artifact found: $HOME/Library/Caches/com.apple.act.mond"
    FOUND=1
    clean=0
  fi

  local hidden
  hidden=$(find /tmp -maxdepth 1 -name ".*" -newer /tmp -type f 2>/dev/null)
  if [ -n "$hidden" ]; then
    echo "[ALERT] Suspicious hidden files in /tmp:"
    echo "        $hidden"
    FOUND=1
    clean=0
  fi

  if [ -e "/tmp/ld.py" ]; then
    echo "[CRITICAL] RAT dropper found: /tmp/ld.py"
    FOUND=1
    clean=0
  fi

  if [ "$clean" -eq 1 ]; then
    echo "[OK] No macOS disk artifacts found."
  fi
}

check_persistence() {
  echo "--- Checking macOS persistence mechanisms ---"
  local clean=1

  for rc in "$HOME/.bashrc" "$HOME/.zshrc" "$HOME/.bash_profile" "$HOME/.zprofile"; do
    if [ -f "$rc" ]; then
      local suspicious
      suspicious=$(grep -n -E 'sfrclak|act\.mond|plain-crypto|nohup.*node|curl.*\|.*sh|wget.*\|.*sh' "$rc" 2>/dev/null)
      if [ -n "$suspicious" ]; then
        echo "[CRITICAL] Suspicious entry in $rc:"
        echo "           $suspicious"
        FOUND=1
        clean=0
      fi
    fi
  done

  local plist_dirs=("$HOME/Library/LaunchAgents" "/Library/LaunchAgents" "/Library/LaunchDaemons")
  for pdir in "${plist_dirs[@]}"; do
    if [ -d "$pdir" ]; then
      local suspicious_plist
      suspicious_plist=$(grep -rl -E 'sfrclak|act\.mond|plain-crypto|wt\.exe' "$pdir" 2>/dev/null)
      if [ -n "$suspicious_plist" ]; then
        echo "[CRITICAL] Suspicious LaunchAgent/Daemon plist:"
        echo "           $suspicious_plist"
        FOUND=1
        clean=0
      fi
    fi
  done

  local cron_entries
  cron_entries=$(crontab -l 2>/dev/null | grep -E 'sfrclak|act\.mond|plain-crypto|nohup.*node')
  if [ -n "$cron_entries" ]; then
    echo "[CRITICAL] Suspicious crontab entries:"
    echo "           $cron_entries"
    FOUND=1
    clean=0
  fi

  if [ "$clean" -eq 1 ]; then
    echo "[OK] No persistence mechanisms found."
  fi
}

check_network() {
  echo "--- Checking macOS network activity ---"
  local clean=1

  local c2_conn
  c2_conn=$(lsof -i -n -P 2>/dev/null | grep -E '142\.11\.206\.73|:8000' | grep -i 'ESTABLISHED')
  if [ -n "$c2_conn" ]; then
    echo "[CRITICAL] Active connection to C2 server detected:"
    echo "           $c2_conn"
    FOUND=1
    clean=0
  fi

  local dns_check
  dns_check=$(log show --predicate 'process == "mDNSResponder"' --style compact --last 1h 2>/dev/null | grep -i 'sfrclak' | head -5)
  if [ -n "$dns_check" ]; then
    echo "[CRITICAL] DNS resolution for C2 domain sfrclak.com detected:"
    echo "           $dns_check"
    FOUND=1
    clean=0
  fi

  local suspect_node
  suspect_node=$(ps aux 2>/dev/null | grep -E 'nohup.*node|node.*sfrclak|node.*act\.mond' | grep -v grep)
  if [ -n "$suspect_node" ]; then
    echo "[CRITICAL] Suspicious node process running:"
    echo "           $suspect_node"
    FOUND=1
    clean=0
  fi

  local suspect_python
  suspect_python=$(ps aux 2>/dev/null | grep -E 'python.*ld\.py|python.*sfrclak' | grep -v grep)
  if [ -n "$suspect_python" ]; then
    echo "[CRITICAL] Suspicious python process running:"
    echo "           $suspect_python"
    FOUND=1
    clean=0
  fi

  if [ "$clean" -eq 1 ]; then
    echo "[OK] No suspicious network activity found."
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

check_disk_artifacts
echo ""

check_persistence
echo ""

check_network
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
