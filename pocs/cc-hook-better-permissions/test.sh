#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PY="$SCRIPT_DIR/permissions.py"

run_test() {
    local label="$1"
    local input="$2"
    local result
    result=$(echo "$input" | python3 "$PY")
    echo "[$label] $result"
}

run_test "curl=ASK" '{"tool_name":"Bash","tool_input":{"command":"curl https://google.com"}}'
run_test "wget=ASK" '{"tool_name":"Bash","tool_input":{"command":"wget http://foo.com/file"}}'
run_test "rm=ASK" '{"tool_name":"Bash","tool_input":{"command":"rm -rf /tmp/test"}}'
run_test "rmdir=ASK" '{"tool_name":"Bash","tool_input":{"command":"rmdir /tmp/old"}}'
run_test "ssh-key=BLOCK" '{"tool_name":"Read","tool_input":{"file_path":"'$HOME'/.ssh/id_rsa"}}'
run_test "env-file=BLOCK" '{"tool_name":"Read","tool_input":{"file_path":"/app/.env"}}'
run_test "aws-creds=BLOCK" '{"tool_name":"Read","tool_input":{"file_path":"'$HOME'/.aws/credentials"}}'
run_test "kube=BLOCK" '{"tool_name":"Read","tool_input":{"file_path":"'$HOME'/.kube/config"}}'
run_test "safe-read=ALLOW" '{"tool_name":"Read","tool_input":{"file_path":"/tmp/readme.txt"}}'
run_test "safe-bash=ALLOW" '{"tool_name":"Bash","tool_input":{"command":"ls -la"}}'
run_test "safe-glob=ALLOW" '{"tool_name":"Glob","tool_input":{"pattern":"src/**/*.py"}}'
run_test "safe-grep=ALLOW" '{"tool_name":"Grep","tool_input":{"pattern":"TODO","path":"/tmp"}}'
run_test "safe-edit=ALLOW" '{"tool_name":"Edit","tool_input":{"file_path":"/tmp/foo.txt"}}'
