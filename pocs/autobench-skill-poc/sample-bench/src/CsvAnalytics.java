import java.io.*;
import java.lang.foreign.*;
import java.nio.channels.*;
import java.util.*;

public class CsvAnalytics {
    static final String[] CATEGORIES = {"books", "clothing", "electronics", "food", "health", "home", "sports", "toys"};
    static final String[] REGIONS = {"east", "north", "south", "west"};

    public static void main(String[] args) throws Exception {
        String filename = args.length > 0 ? args[0] : "data.csv";

        try (var arena = Arena.ofShared();
             FileChannel channel = new RandomAccessFile(filename, "r").getChannel()) {

            long fileSize = channel.size();
            MemorySegment segment = channel.map(FileChannel.MapMode.READ_ONLY, 0, fileSize, arena);

            long headerEnd = 0;
            while (headerEnd < fileSize && segment.get(ValueLayout.JAVA_BYTE, headerEnd) != '\n') headerEnd++;
            headerEnd++;

            int numThreads = Runtime.getRuntime().availableProcessors();
            long dataSize = fileSize - headerEnd;
            long chunkSize = dataSize / numThreads;

            long[] chunkStarts = new long[numThreads];
            long[] chunkEnds = new long[numThreads];

            chunkStarts[0] = headerEnd;
            for (int t = 1; t < numThreads; t++) {
                long pos = headerEnd + t * chunkSize;
                while (pos < fileSize && segment.get(ValueLayout.JAVA_BYTE, pos) != '\n') pos++;
                pos++;
                chunkStarts[t] = pos;
                chunkEnds[t - 1] = chunkStarts[t];
            }
            chunkEnds[numThreads - 1] = fileSize;

            PartialResult[] results = new PartialResult[numThreads];
            Thread[] threads = new Thread[numThreads];

            for (int t = 0; t < numThreads; t++) {
                final int idx = t;
                final long start = chunkStarts[t];
                final long end = chunkEnds[t];
                threads[t] = Thread.ofVirtual().start(() -> {
                    results[idx] = processChunk(segment, start, end);
                });
            }
            for (Thread thread : threads) thread.join();

            long totalRows = 0;
            double totalRevenue = 0.0;
            double minPrice = Double.MAX_VALUE;
            double maxPrice = Double.MIN_VALUE;
            long totalQuantity = 0;
            double[] revenueByCategory = new double[CATEGORIES.length];
            double[] revenueByRegion = new double[REGIONS.length];
            long[] countByCategory = new long[CATEGORIES.length];

            for (PartialResult r : results) {
                totalRows += r.totalRows;
                totalRevenue += r.totalRevenue;
                totalQuantity += r.totalQuantity;
                if (r.minPrice < minPrice) minPrice = r.minPrice;
                if (r.maxPrice > maxPrice) maxPrice = r.maxPrice;
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
            System.out.println("Total revenue: " + String.format("%.2f", totalRevenue));
            System.out.println("Total quantity: " + totalQuantity);
            System.out.println("Min price: " + String.format("%.2f", minPrice));
            System.out.println("Max price: " + String.format("%.2f", maxPrice));
            System.out.println("Avg revenue per row: " + String.format("%.2f", totalRevenue / totalRows));
            System.out.println();
            System.out.println("Revenue by category:");
            Integer[] catIdx = new Integer[CATEGORIES.length];
            for (int i = 0; i < catIdx.length; i++) catIdx[i] = i;
            Arrays.sort(catIdx, (a, b) -> Double.compare(revenueByCategory[b], revenueByCategory[a]));
            for (int i : catIdx) System.out.println("  " + CATEGORIES[i] + ": " + String.format("%.2f", revenueByCategory[i]));
            System.out.println();
            System.out.println("Revenue by region:");
            Integer[] regIdx = new Integer[REGIONS.length];
            for (int i = 0; i < regIdx.length; i++) regIdx[i] = i;
            Arrays.sort(regIdx, (a, b) -> Double.compare(revenueByRegion[b], revenueByRegion[a]));
            for (int i : regIdx) System.out.println("  " + REGIONS[i] + ": " + String.format("%.2f", revenueByRegion[i]));
            System.out.println();
            System.out.println("Count by category:");
            Arrays.sort(catIdx, (a, b) -> Long.compare(countByCategory[b], countByCategory[a]));
            for (int i : catIdx) System.out.println("  " + CATEGORIES[i] + ": " + countByCategory[i]);
        }
    }

    static PartialResult processChunk(MemorySegment seg, long start, long end) {
        PartialResult result = new PartialResult();
        byte[] lineBytes = new byte[256];
        long pos = start;

        while (pos < end) {
            int lineLen = 0;
            while (pos < end) {
                byte b = seg.get(ValueLayout.JAVA_BYTE, pos++);
                if (b == '\n') break;
                if (lineLen < lineBytes.length) lineBytes[lineLen++] = b;
            }
            if (lineLen == 0) continue;

            int p0 = -1, p1 = -1, p2 = -1, p3 = -1, p4 = -1;
            int commaCount = 0;
            for (int i = 0; i < lineLen; i++) {
                if (lineBytes[i] == ',') {
                    commaCount++;
                    switch (commaCount) {
                        case 1: p0 = i; break;
                        case 2: p1 = i; break;
                        case 3: p2 = i; break;
                        case 4: p3 = i; break;
                        case 5: p4 = i; break;
                    }
                }
            }
            if (p4 == -1) continue;

            int catOrd = matchCategory(lineBytes, p0 + 1, p1);
            int regOrd = matchRegion(lineBytes, p1 + 1, p2);
            int quantity = parseIntBytes(lineBytes, p2 + 1, p3);
            double price = parseDoubleBytes(lineBytes, p3 + 1, p4);
            double discount = parseDoubleBytes(lineBytes, p4 + 1, lineLen);

            double revenue = quantity * price * (1.0 - discount);
            result.totalRows++;
            result.totalRevenue += revenue;
            result.totalQuantity += quantity;
            if (price < result.minPrice) result.minPrice = price;
            if (price > result.maxPrice) result.maxPrice = price;

            if (catOrd >= 0) {
                result.revenueByCategory[catOrd] += revenue;
                result.countByCategory[catOrd]++;
            }
            if (regOrd >= 0) {
                result.revenueByRegion[regOrd] += revenue;
            }
        }
        return result;
    }

    static int matchCategory(byte[] b, int from, int to) {
        int len = to - from;
        byte first = b[from];
        return switch (first) {
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

    static int matchRegion(byte[] b, int from, int to) {
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

    static double parseDoubleBytes(byte[] b, int from, int to) {
        long intPart = 0;
        int i = from;
        while (i < to && b[i] != '.') {
            intPart = intPart * 10 + (b[i] - '0');
            i++;
        }
        if (i >= to) return intPart;
        i++;
        long fracPart = 0;
        int fracDigits = 0;
        while (i < to) {
            fracPart = fracPart * 10 + (b[i] - '0');
            fracDigits++;
            i++;
        }
        double divisor = 1.0;
        for (int d = 0; d < fracDigits; d++) divisor *= 10.0;
        return intPart + fracPart / divisor;
    }

    static class PartialResult {
        long totalRows = 0;
        double totalRevenue = 0.0;
        double minPrice = Double.MAX_VALUE;
        double maxPrice = Double.MIN_VALUE;
        long totalQuantity = 0;
        double[] revenueByCategory = new double[CATEGORIES.length];
        double[] revenueByRegion = new double[REGIONS.length];
        long[] countByCategory = new long[CATEGORIES.length];
    }
}
