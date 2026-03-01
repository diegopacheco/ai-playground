#!/usr/bin/env bash
set -euo pipefail

BINARY="$1"
DATA_DIR="${2:-./data}"
RUNS="${3:-3}"

echo "=== AutoBench: CSV Analytics ==="
echo "Binary: $BINARY"
echo "Data dir: $DATA_DIR"
echo "Runs: $RUNS"
echo ""

echo "--- Correctness Check ---"
SAMPLE_OUTPUT=$($BINARY "$DATA_DIR" --validate 2>&1)
if [ $? -ne 0 ]; then
    echo "FAIL: Correctness check failed"
    echo "$SAMPLE_OUTPUT"
    exit 1
fi
echo "PASS: Correctness validated"
echo ""

echo "--- Benchmark Results ---"
TOTAL_TIME=0
TOTAL_THROUGHPUT=0
TOTAL_MEMORY=0

for i in $(seq 1 $RUNS); do
    echo "Run $i/$RUNS:"

    START_TIME=$(date +%s%N)
    OUTPUT=$($BINARY "$DATA_DIR" 2>&1)
    END_TIME=$(date +%s%N)

    ELAPSED_MS=$(( (END_TIME - START_TIME) / 1000000 ))
    ELAPSED_S=$(echo "scale=3; $ELAPSED_MS / 1000" | bc)

    MEMORY_KB=$(ps -o rss= -p $$ 2>/dev/null || echo "0")

    FILE_COUNT=$(find "$DATA_DIR" -name "*.csv" | wc -l | tr -d ' ')
    THROUGHPUT=$(echo "scale=2; $FILE_COUNT / $ELAPSED_S" | bc)

    echo "  Time: ${ELAPSED_S}s"
    echo "  Throughput: ${THROUGHPUT} files/s"
    echo "  Memory: ${MEMORY_KB} KB"
    echo ""

    TOTAL_TIME=$(echo "$TOTAL_TIME + $ELAPSED_S" | bc)
    TOTAL_THROUGHPUT=$(echo "$TOTAL_THROUGHPUT + $THROUGHPUT" | bc)
    TOTAL_MEMORY=$(echo "$TOTAL_MEMORY + $MEMORY_KB" | bc)
done

AVG_TIME=$(echo "scale=3; $TOTAL_TIME / $RUNS" | bc)
AVG_THROUGHPUT=$(echo "scale=2; $TOTAL_THROUGHPUT / $RUNS" | bc)
AVG_MEMORY=$(echo "scale=0; $TOTAL_MEMORY / $RUNS" | bc)

echo "--- Average Results ---"
echo "Avg Time: ${AVG_TIME}s"
echo "Avg Throughput: ${AVG_THROUGHPUT} files/s"
echo "Avg Memory: ${AVG_MEMORY} KB"
echo ""
echo "=== AutoBench Complete ==="
