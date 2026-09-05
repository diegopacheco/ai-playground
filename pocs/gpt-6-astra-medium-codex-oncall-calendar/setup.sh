set -eu
cd "$(dirname "$0")"
for tool in bun python3.14 npx curl; do
  command -v "$tool" >/dev/null || { echo "Required command missing: $tool" >&2; exit 1; }
done
test "$(python3.14 -c 'import platform; print(platform.python_version())')" = "3.14.6" || { echo 'Python 3.14.6 is required.' >&2; exit 1; }
test "$(bun --version)" = "1.4.0" || { echo 'Bun 1.4.0 is required.' >&2; exit 1; }
bun install --frozen-lockfile
