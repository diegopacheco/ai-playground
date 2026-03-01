import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;

public class CsvAnalytics {
    static final String[] CATEGORIES = {"books", "clothing", "electronics", "food", "health", "home", "sports", "toys"};
    static final String[] REGIONS = {"east", "north", "south", "west"};
    static final int CHUNK_SIZE = 1 << 16;

    public static void main(String[] args) throws Exception {
        String filename = args.length > 0 ? args[0] : "data.csv";

        try (RandomAccessFile raf = new RandomAccessFile(filename, "r");
             FileChannel channel = raf.getChannel()) {

            long fileSize = channel.size();
            MappedByteBuffer buffer = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize);

            int headerEnd = 0;
            while (headerEnd < fileSize && buffer.get(headerEnd) != '\n') headerEnd++;
            headerEnd++;

            int numThreads = Runtime.getRuntime().availableProcessors();
            long dataSize = fileSize - headerEnd;
            long chunkSize = dataSize / numThreads;

            long[] chunkStarts = new long[numThreads];
            long[] chunkEnds = new long[numThreads];

            chunkStarts[0] = headerEnd;
            for (int t = 1; t < numThreads; t++) {
                long pos = headerEnd + t * chunkSize;
                while (pos < fileSize && buffer.get((int) pos) != '\n') pos++;
                pos++;
                chunkStarts[t] = pos;
                chunkEnds[t - 1] = chunkStarts[t];
            }
            chunkEnds[numThreads - 1] = fileSize;

            PartialResult[] results = new PartialResult[numThreads];
            Thread[] threads = new Thread[numThreads];

            for (int t = 0; t < numThreads; t++) {
                final int idx = t;
                final int start = (int) chunkStarts[t];
                final int end = (int) chunkEnds[t];
                threads[t] = new Thread(() -> {
                    results[idx] = processChunk(buffer, start, end);
                });
                threads[t].start();
            }
            for (Thread thread : threads) thread.join();

            long totalRows = 0;
            long totalRevenueFP = 0;
            long minPriceFP = Long.MAX_VALUE;
            long maxPriceFP = Long.MIN_VALUE;
            long totalQuantity = 0;
            long[] revenueByCategory = new long[CATEGORIES.length];
            long[] revenueByRegion = new long[REGIONS.length];
            long[] countByCategory = new long[CATEGORIES.length];

            for (PartialResult r : results) {
                totalRows += r.totalRows;
                totalRevenueFP += r.totalRevenueFP;
                totalQuantity += r.totalQuantity;
                if (r.minPriceFP < minPriceFP) minPriceFP = r.minPriceFP;
                if (r.maxPriceFP > maxPriceFP) maxPriceFP = r.maxPriceFP;
                for (int i = 0; i < CATEGORIES.length; i++) {
                    revenueByCategory[i] += r.revenueByCategory[i];
                    countByCategory[i] += r.countByCategory[i];
                }
                for (int i = 0; i < REGIONS.length; i++) {
                    revenueByRegion[i] += r.revenueByRegion[i];
                }
            }

            System.out.println("=== CSV Analytics Results ===");
            System.out.println("Total rows: " + totalRows);
            System.out.println("Total revenue: " + fpToString(totalRevenueFP));
            System.out.println("Total quantity: " + totalQuantity);
            System.out.println("Min price: " + fpToString(minPriceFP));
            System.out.println("Max price: " + fpToString(maxPriceFP));
            System.out.println("Avg revenue per row: " + String.format("%.2f", (double) totalRevenueFP / 100.0 / totalRows));
            System.out.println();
            System.out.println("Revenue by category:");
            Integer[] catIdx = new Integer[CATEGORIES.length];
            for (int i = 0; i < catIdx.length; i++) catIdx[i] = i;
            Arrays.sort(catIdx, (a, b) -> Long.compare(revenueByCategory[b], revenueByCategory[a]));
            for (int i : catIdx) System.out.println("  " + CATEGORIES[i] + ": " + fpToString(revenueByCategory[i]));
            System.out.println();
            System.out.println("Revenue by region:");
            Integer[] regIdx = new Integer[REGIONS.length];
            for (int i = 0; i < regIdx.length; i++) regIdx[i] = i;
            Arrays.sort(regIdx, (a, b) -> Long.compare(revenueByRegion[b], revenueByRegion[a]));
            for (int i : regIdx) System.out.println("  " + REGIONS[i] + ": " + fpToString(revenueByRegion[i]));
            System.out.println();
            System.out.println("Count by category:");
            Arrays.sort(catIdx, (a, b) -> Long.compare(countByCategory[b], countByCategory[a]));
            for (int i : catIdx) System.out.println("  " + CATEGORIES[i] + ": " + countByCategory[i]);
        }
    }

    static String fpToString(long fp) {
        long whole = fp / 10000;
        long frac = Math.abs(fp % 10000);
        return String.format("%d.%02d", whole, frac / 100);
    }

    static PartialResult processChunk(MappedByteBuffer masterBuffer, int start, int end) {
        PartialResult result = new PartialResult();
        byte[] bulk = new byte[CHUNK_SIZE];
        int pos = start;

        byte[] lineBytes = new byte[256];
        int lineLen = 0;

        while (pos < end) {
            int toRead = Math.min(CHUNK_SIZE, end - pos);
            ByteBuffer slice = masterBuffer.duplicate();
            slice.position(pos);
            slice.limit(pos + toRead);
            slice.get(bulk, 0, toRead);
            pos += toRead;

            for (int i = 0; i < toRead; i++) {
                byte b = bulk[i];
                if (b == '\n') {
                    if (lineLen > 0) processLine(lineBytes, lineLen, result);
                    lineLen = 0;
                } else {
                    if (lineLen < lineBytes.length) lineBytes[lineLen++] = b;
                }
            }
        }
        if (lineLen > 0) processLine(lineBytes, lineLen, result);
        return result;
    }

    static void processLine(byte[] b, int len, PartialResult result) {
        int c0 = 0, c1 = 0, c2 = 0, c3 = 0, c4 = 0;
        int found = 0;
        for (int i = 0; i < len && found < 5; i++) {
            if (b[i] == ',') {
                found++;
                switch (found) {
                    case 1: c0 = i; break;
                    case 2: c1 = i; break;
                    case 3: c2 = i; break;
                    case 4: c3 = i; break;
                    case 5: c4 = i; break;
                }
            }
        }
        if (found < 5) return;

        int catOrd = matchCategory(b, c0 + 1, c1);
        int regOrd = matchRegion(b, c1 + 1);
        int quantity = parseIntBytes(b, c2 + 1, c3);
        long priceFP = parseFP(b, c3 + 1, c4);
        long discountFP = parseFP(b, c4 + 1, len);

        long revenueFP = (quantity * priceFP * (10000 - discountFP)) / 10000;
        result.totalRows++;
        result.totalRevenueFP += revenueFP;
        result.totalQuantity += quantity;
        if (priceFP < result.minPriceFP) result.minPriceFP = priceFP;
        if (priceFP > result.maxPriceFP) result.maxPriceFP = priceFP;

        if (catOrd >= 0) {
            result.revenueByCategory[catOrd] += revenueFP;
            result.countByCategory[catOrd]++;
        }
        if (regOrd >= 0) {
            result.revenueByRegion[regOrd] += revenueFP;
        }
    }

    static long parseFP(byte[] b, int from, int to) {
        long intPart = 0;
        int i = from;
        while (i < to && b[i] != '.') {
            intPart = intPart * 10 + (b[i] - '0');
            i++;
        }
        if (i >= to) return intPart * 10000;
        i++;
        long frac = 0;
        int digits = 0;
        while (i < to && digits < 4) {
            frac = frac * 10 + (b[i] - '0');
            digits++;
            i++;
        }
        while (digits < 4) { frac *= 10; digits++; }
        return intPart * 10000 + frac;
    }

    static int matchCategory(byte[] b, int from, int to) {
        int len = to - from;
        return switch (b[from]) {
            case 'b' -> 0;
            case 'c' -> 1;
            case 'e' -> 2;
            case 'f' -> 3;
            case 'h' -> len == 6 ? 4 : 5;
            case 's' -> 6;
            case 't' -> 7;
            default -> -1;
        };
    }

    static int matchRegion(byte[] b, int from) {
        return switch (b[from]) {
            case 'e' -> 0;
            case 'n' -> 1;
            case 's' -> 2;
            case 'w' -> 3;
            default -> -1;
        };
    }

    static int parseIntBytes(byte[] b, int from, int to) {
        int result = 0;
        for (int i = from; i < to; i++) {
            result = result * 10 + (b[i] - '0');
        }
        return result;
    }

    static class PartialResult {
        long totalRows = 0;
        long totalRevenueFP = 0;
        long minPriceFP = Long.MAX_VALUE;
        long maxPriceFP = Long.MIN_VALUE;
        long totalQuantity = 0;
        long[] revenueByCategory = new long[CATEGORIES.length];
        long[] revenueByRegion = new long[REGIONS.length];
        long[] countByCategory = new long[CATEGORIES.length];
    }
}
