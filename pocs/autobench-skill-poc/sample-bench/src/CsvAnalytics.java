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
        Map<String, Double> revenueByCategory = new HashMap<>();
        Map<String, Double> revenueByRegion = new HashMap<>();
        Map<String, Long> countByCategory = new HashMap<>();

        try (BufferedReader reader = new BufferedReader(new FileReader(filename))) {
            String header = reader.readLine();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] parts = line.split(",");
                String category = parts[1];
                String region = parts[2];
                int quantity = Integer.parseInt(parts[3]);
                double price = Double.parseDouble(parts[4]);
                double discount = Double.parseDouble(parts[5]);

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
}
