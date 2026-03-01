import java.io.*;
import java.nio.*;
import java.nio.channels.*;
import java.util.*;
import java.util.concurrent.*;

public class CsvAnalytics {
    static final Map<String, String> INTERN_CACHE = new ConcurrentHashMap<>(32);

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

            ExecutorService pool = Executors.newFixedThreadPool(numThreads);
            Future<PartialResult>[] futures = new Future[numThreads];

            for (int t = 0; t < numThreads; t++) {
                final int start = (int) chunkStarts[t];
                final int end = (int) chunkEnds[t];
                futures[t] = pool.submit(() -> processChunk(buffer, start, end));
            }

            long totalRows = 0;
            double totalRevenue = 0.0;
            double minPrice = Double.MAX_VALUE;
            double maxPrice = Double.MIN_VALUE;
            long totalQuantity = 0;
            HashMap<String, Double> revenueByCategory = new HashMap<>(16);
            HashMap<String, Double> revenueByRegion = new HashMap<>(8);
            HashMap<String, Long> countByCategory = new HashMap<>(16);

            for (Future<PartialResult> f : futures) {
                PartialResult r = f.get();
                totalRows += r.totalRows;
                totalRevenue += r.totalRevenue;
                totalQuantity += r.totalQuantity;
                if (r.minPrice < minPrice) minPrice = r.minPrice;
                if (r.maxPrice > maxPrice) maxPrice = r.maxPrice;
                r.revenueByCategory.forEach((k, v) -> revenueByCategory.merge(k, v, Double::sum));
                r.revenueByRegion.forEach((k, v) -> revenueByRegion.merge(k, v, Double::sum));
                r.countByCategory.forEach((k, v) -> countByCategory.merge(k, v, Long::sum));
            }

            pool.shutdown();

            System.out.println("=== CSV Analytics Results ===");
            System.out.println("Total rows: " + totalRows);
            System.out.println("Total revenue: " + String.format("%.2f", totalRevenue));
            System.out.println("Total quantity: " + totalQuantity);
            System.out.println("Min price: " + String.format("%.2f", minPrice));
            System.out.println("Max price: " + String.format("%.2f", maxPrice));
            System.out.println("Avg revenue per row: " + String.format("%.2f", totalRevenue / totalRows));
            System.out.println();
            System.out.println("Revenue by category:");
            revenueByCategory.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(e -> System.out.println("  " + e.getKey() + ": " + String.format("%.2f", e.getValue())));
            System.out.println();
            System.out.println("Revenue by region:");
            revenueByRegion.entrySet().stream()
                .sorted(Map.Entry.<String, Double>comparingByValue().reversed())
                .forEach(e -> System.out.println("  " + e.getKey() + ": " + String.format("%.2f", e.getValue())));
            System.out.println();
            System.out.println("Count by category:");
            countByCategory.entrySet().stream()
                .sorted(Map.Entry.<String, Long>comparingByValue().reversed())
                .forEach(e -> System.out.println("  " + e.getKey() + ": " + e.getValue()));
        }
    }

    static PartialResult processChunk(MappedByteBuffer masterBuffer, int start, int end) {
        ByteBuffer buf = masterBuffer.duplicate();
        buf.position(start);
        buf.limit(end);

        PartialResult result = new PartialResult();
        byte[] lineBytes = new byte[256];

        while (buf.hasRemaining()) {
            int lineLen = 0;
            while (buf.hasRemaining()) {
                byte b = buf.get();
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

            String category = intern(new String(lineBytes, p0 + 1, p1 - p0 - 1));
            String region = intern(new String(lineBytes, p1 + 1, p2 - p1 - 1));
            int quantity = parseIntBytes(lineBytes, p2 + 1, p3);
            double price = parseDoubleBytes(lineBytes, p3 + 1, p4);
            double discount = parseDoubleBytes(lineBytes, p4 + 1, lineLen);

            double revenue = quantity * price * (1.0 - discount);
            result.totalRows++;
            result.totalRevenue += revenue;
            result.totalQuantity += quantity;
            if (price < result.minPrice) result.minPrice = price;
            if (price > result.maxPrice) result.maxPrice = price;

            result.revenueByCategory.merge(category, revenue, Double::sum);
            result.revenueByRegion.merge(region, revenue, Double::sum);
            result.countByCategory.merge(category, 1L, Long::sum);
        }
        return result;
    }

    static String intern(String s) {
        return INTERN_CACHE.computeIfAbsent(s, k -> k);
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
        HashMap<String, Double> revenueByCategory = new HashMap<>(16);
        HashMap<String, Double> revenueByRegion = new HashMap<>(8);
        HashMap<String, Long> countByCategory = new HashMap<>(16);
    }
}
