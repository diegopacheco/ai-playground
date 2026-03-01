import java.io.*;
import java.util.*;
import jdk.incubator.vector.*;

public class CsvAnalytics {
    static final String[] CATEGORIES = {"books", "clothing", "electronics", "food", "health", "home", "sports", "toys"};
    static final String[] REGIONS = {"east", "north", "south", "west"};
    static final int BULK_SIZE = 1 << 18;
    static final ByteVector COMMA_VEC = ByteVector.broadcast(ByteVector.SPECIES_256, (byte) ',');
    static final ByteVector NEWLINE_VEC = ByteVector.broadcast(ByteVector.SPECIES_256, (byte) '\n');

    public static void main(String[] args) throws Exception {
        String filename = args.length > 0 ? args[0] : "data.csv";

        File f = new File(filename);
        long fileSize = f.length();
        byte[] fileData = new byte[(int) fileSize];

        try (FileInputStream fis = new FileInputStream(f)) {
            int offset = 0;
            while (offset < fileSize) {
                int read = fis.read(fileData, offset, (int) fileSize - offset);
                if (read < 0) break;
                offset += read;
            }
        }

        int headerEnd = 0;
        while (headerEnd < fileSize && fileData[headerEnd] != '\n') headerEnd++;
        headerEnd++;

        int numThreads = Runtime.getRuntime().availableProcessors();
        int dataSize = (int) fileSize - headerEnd;
        int chunkSize = dataSize / numThreads;

        int[] chunkStarts = new int[numThreads];
        int[] chunkEnds = new int[numThreads];

        chunkStarts[0] = headerEnd;
        for (int t = 1; t < numThreads; t++) {
            int pos = headerEnd + t * chunkSize;
            while (pos < fileSize && fileData[pos] != '\n') pos++;
            pos++;
            chunkStarts[t] = pos;
            chunkEnds[t - 1] = chunkStarts[t];
        }
        chunkEnds[numThreads - 1] = (int) fileSize;

        PartialResult[] results = new PartialResult[numThreads];
        Thread[] threads = new Thread[numThreads];

        for (int t = 0; t < numThreads; t++) {
            final int idx = t;
            final int start = chunkStarts[t];
            final int end = chunkEnds[t];
            threads[t] = new Thread(() -> {
                results[idx] = processChunk(fileData, start, end);
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
        System.out.println("Avg revenue per row: " + String.format("%.2f", (double) totalRevenueFP / 10000.0 / totalRows));
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

    static String fpToString(long fp) {
        long whole = fp / 10000;
        long frac = Math.abs(fp % 10000);
        return String.format("%d.%02d", whole, frac / 100);
    }

    static PartialResult processChunk(byte[] data, int start, int end) {
        PartialResult result = new PartialResult();
        int lineStart = start;

        for (int pos = start; pos < end; pos++) {
            if (data[pos] == '\n') {
                int lineLen = pos - lineStart;
                if (lineLen > 0) processLine(data, lineStart, lineLen, result);
                lineStart = pos + 1;
            }
        }
        if (lineStart < end) processLine(data, lineStart, end - lineStart, result);
        return result;
    }

    static void processLine(byte[] data, int off, int len, PartialResult result) {
        int end = off + len;
        int c0 = findComma(data, off, end);
        if (c0 < 0) return;
        int c1 = findComma(data, c0 + 1, end);
        if (c1 < 0) return;
        int c2 = findComma(data, c1 + 1, end);
        if (c2 < 0) return;
        int c3 = findComma(data, c2 + 1, end);
        if (c3 < 0) return;
        int c4 = findComma(data, c3 + 1, end);
        if (c4 < 0) return;

        int catOrd = matchCategory(data, c0 + 1, c1);
        int regOrd = matchRegion(data, c1 + 1);
        int quantity = parseIntDirect(data, c2 + 1, c3);
        long priceFP = parseFP(data, c3 + 1, c4);
        long discountFP = parseFP(data, c4 + 1, end);

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

    static int findComma(byte[] data, int from, int end) {
        int i = from;
        int vecLen = ByteVector.SPECIES_256.length();
        int limit = end - vecLen;
        while (i <= limit) {
            ByteVector v = ByteVector.fromArray(ByteVector.SPECIES_256, data, i);
            long mask = v.eq(COMMA_VEC).toLong();
            if (mask != 0) return i + Long.numberOfTrailingZeros(mask);
            i += vecLen;
        }
        while (i < end) {
            if (data[i] == ',') return i;
            i++;
        }
        return -1;
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

    static int parseIntDirect(byte[] b, int from, int to) {
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
