#!/bin/bash

SAFEHOUSE_DIR="./agent-safehouse"

if [ ! -d "$SAFEHOUSE_DIR" ]; then
  git clone https://github.com/eugene1g/agent-safehouse.git "$SAFEHOUSE_DIR"
fi

SAFEHOUSE="$SAFEHOUSE_DIR/bin/safehouse.sh"
POLICY=$($SAFEHOUSE)

echo "=== Agent Safehouse POC ==="
echo "Policy file: $POLICY"
echo ""

echo "[1] SAFE: Reading weather info (allowed - simple echo command)"
sandbox-exec -f "$POLICY" -- /bin/echo "The weather in Porto Alegre, Brazil is typically warm and humid, around 25-30C"
echo "Exit code: $?"
echo ""

echo "[2] SAFE: Listing current directory (allowed - workdir is readable)"
sandbox-exec -f "$POLICY" -- /bin/ls .
echo "Exit code: $?"
echo ""

echo "[3] DENIED: Trying to read SSH private key (blocked by sandbox)"
sandbox-exec -f "$POLICY" -- /bin/cat "$HOME/.ssh/id_ed25519" 2>&1
echo "Exit code: $?"
echo ""

echo "[4] DENIED: Trying to read /etc/shadow (blocked by sandbox)"
sandbox-exec -f "$POLICY" -- /bin/cat /etc/master.passwd 2>&1
echo "Exit code: $?"
echo ""

echo "[5] SAFE: known_hosts is allowed (git-over-ssh needs it)"
sandbox-exec -f "$POLICY" -- /bin/cat "$HOME/.ssh/known_hosts" 2>&1 | head -2
echo "Exit code: $?"
echo ""

echo "=== Done ==="
