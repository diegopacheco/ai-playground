set -eu
cd "$(dirname "$0")"
bash setup.sh
npx playwright install chromium
bun run typecheck
bun run test
bun run build
npx playwright test
