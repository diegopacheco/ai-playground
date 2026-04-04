#!/bin/bash
KEYS=$(podman exec redis-fs redis-cli KEYS '*')
if [ -z "$KEYS" ]; then
  echo "No keys found"
  exit 0
fi
for KEY in $KEYS; do
  TYPE=$(podman exec redis-fs redis-cli TYPE "$KEY" | tr -d '[:space:]')
  echo "=== $KEY ($TYPE) ==="
  case $TYPE in
    string)
      podman exec redis-fs redis-cli GET "$KEY"
      ;;
    hash)
      podman exec redis-fs redis-cli HGETALL "$KEY"
      ;;
    set)
      podman exec redis-fs redis-cli SMEMBERS "$KEY"
      ;;
    list)
      podman exec redis-fs redis-cli LRANGE "$KEY" 0 -1
      ;;
    *)
      echo "(unsupported type)"
      ;;
  esac
  echo ""
done
