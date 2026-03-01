import java.io.*;
import java.util.*;

public class CsvAnalytics {
    public static void main(String[] args) throws Exception {
        String filename = args.length > 0 ? args[0] : "data.csv";

        long totalRows = 0;
        double totalRevenue = 0.0;
        double minPrice = Double.MAX_VALUE;
        double maxPrice = Double.MIN_VALUE;
        long totalQuantity = 0;
        HashMap<String, Double> revenueByCategory = new HashMap<>(16);
        HashMap<String, Double> revenueByRegion = new HashMap<>(8);
        HashMap<String, Long> countByCategory = new HashMap<>(16);

        try (BufferedReader reader = new BufferedReader(new FileReader(filename), 1 << 20)) {
            reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                int p0 = line.indexOf(',');
                int p1 = line.indexOf(',', p0 + 1);
                int p2 = line.indexOf(',', p1 + 1);
                int p3 = line.indexOf(',', p2 + 1);
                int p4 = line.indexOf(',', p3 + 1);

                String category = line.substring(p0 + 1, p1);
                String region = line.substring(p1 + 1, p2);
                int quantity = parseIntFast(line, p2 + 1, p3);
                double price = parseDoubleFast(line, p3 + 1, p4);
                double discount = parseDoubleFast(line, p4 + 1, line.length());

                double revenue = quantity * price * (1.0 - discount);
                totalRows++;
                totalRevenue += revenue;
                totalQuantity += quantity;
                if (price < minPrice) minPrice = price;
                if (price > maxPrice) maxPrice = price;

                revenueByCategory.merge(category, revenue, Double::sum);
                revenueByRegion.merge(region, revenue, Double::sum);
                countByCategory.merge(category, 1L, Long::sum);
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

    static int parseIntFast(String s, int from, int to) {
        int result = 0;
        for (int i = from; i < to; i++) {
            result = result * 10 + (s.charAt(i) - '0');
        }
        return result;
    }

    static double parseDoubleFast(String s, int from, int to) {
        long intPart = 0;
        int i = from;
        while (i < to && s.charAt(i) != '.') {
            intPart = intPart * 10 + (s.charAt(i) - '0');
            i++;
        }
        if (i >= to) return intPart;
        i++;
        long fracPart = 0;
        int fracDigits = 0;
        while (i < to) {
            fracPart = fracPart * 10 + (s.charAt(i) - '0');
            fracDigits++;
            i++;
        }
        double divisor = 1.0;
        for (int d = 0; d < fracDigits; d++) divisor *= 10.0;
        return intPart + fracPart / divisor;
    }
}
