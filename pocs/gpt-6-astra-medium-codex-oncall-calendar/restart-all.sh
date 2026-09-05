set -eu
cd "$(dirname "$0")"
bash stop-all.sh
exec bash start-all.sh
