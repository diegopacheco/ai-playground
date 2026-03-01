#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "=== CSV Analytics Benchmark ==="
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""

javac --add-modules jdk.incubator.vector -d out src/GenerateCSV.java src/CsvAnalytics.java 2>&1 | grep -v "warning:"

if [ ! -f data.csv ]; then
    echo "Generating 1M row CSV..."
    java -cp out GenerateCSV data.csv 1000000
    echo ""
fi

ROW_COUNT=$(wc -l < data.csv | tr -d ' ')
EXPECTED=1000001
if [ "$ROW_COUNT" -ne "$EXPECTED" ]; then
    echo "FAIL: Expected $EXPECTED lines (header + 1M rows), got $ROW_COUNT"
    exit 1
fi
echo "Correctness check: $ROW_COUNT lines (1M data rows + header) OK"

echo ""
echo "Running analytics validation..."
VALIDATION=$(java --add-modules jdk.incubator.vector -cp out CsvAnalytics data.csv 2>&1 | grep -v "WARNING:")
VTOTAL=$(echo "$VALIDATION" | grep "Total rows:" | awk '{print $3}')
if [ "$VTOTAL" != "1000000" ]; then
    echo "FAIL: Analytics reported $VTOTAL rows instead of 1000000"
    exit 1
fi
echo "Validation: $VTOTAL rows processed OK"
echo ""

echo "=== Benchmark Runs ==="
TIMES=()
HEAP_SIZES=()

for RUN in 1 2 3; do
    echo "--- Run $RUN ---"
    START_NS=$(python3 -c "import time; print(int(time.time_ns()))")

    OUTPUT=$(java --add-modules jdk.incubator.vector -cp out -Xmx512m CsvAnalytics data.csv 2>&1 | grep -v "WARNING:")

    END_NS=$(python3 -c "import time; print(int(time.time_ns()))")
    ELAPSED_MS=$(python3 -c "print(($END_NS - $START_NS) / 1_000_000)")
    TIMES+=("$ELAPSED_MS")
    echo "  Time: ${ELAPSED_MS}ms"
    echo "$OUTPUT" | head -3
    echo ""
done

echo "=== Results ==="
AVG=$(python3 -c "
times = [${TIMES[0]}, ${TIMES[1]}, ${TIMES[2]}]
avg = sum(times) / len(times)
print(f'{avg:.1f}')
")

FILE_SIZE_MB=$(python3 -c "
import os
size = os.path.getsize('data.csv')
print(f'{size / (1024*1024):.1f}')
")

THROUGHPUT=$(python3 -c "
avg_sec = $AVG / 1000.0
file_mb = $FILE_SIZE_MB
print(f'{file_mb / avg_sec:.1f}')
")

ROWS_PER_SEC=$(python3 -c "
avg_sec = $AVG / 1000.0
print(f'{1_000_000 / avg_sec:.0f}')
")

echo "Run times: ${TIMES[0]}ms, ${TIMES[1]}ms, ${TIMES[2]}ms"
echo "Average time: ${AVG}ms"
echo "File size: ${FILE_SIZE_MB}MB"
echo "Throughput: ${THROUGHPUT} MB/s"
echo "Rows/sec: ${ROWS_PER_SEC}"
echo ""
echo "METRICS_AVG_MS=$AVG"
echo "METRICS_THROUGHPUT_MBS=$THROUGHPUT"
echo "METRICS_ROWS_PER_SEC=$ROWS_PER_SEC"
