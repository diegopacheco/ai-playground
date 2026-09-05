set -eu
cd "$(dirname "$0")"
bash setup.sh
bun run build
