import java.io.*;
import java.util.Random;

public class GenerateCSV {
    public static void main(String[] args) throws Exception {
        String filename = args.length > 0 ? args[0] : "data.csv";
        int rows = args.length > 1 ? Integer.parseInt(args[1]) : 1_000_000;
        Random rng = new Random(42);
        String[] categories = {"electronics", "clothing", "food", "books", "sports", "home", "toys", "health"};
        String[] regions = {"north", "south", "east", "west"};

        try (BufferedWriter w = new BufferedWriter(new FileWriter(filename), 1 << 16)) {
            w.write("id,category,region,quantity,price,discount");
            w.newLine();
            for (int i = 1; i <= rows; i++) {
                w.write(String.valueOf(i));
                w.write(',');
                w.write(categories[rng.nextInt(categories.length)]);
                w.write(',');
                w.write(regions[rng.nextInt(regions.length)]);
                w.write(',');
                w.write(String.valueOf(rng.nextInt(100) + 1));
                w.write(',');
                w.write(String.format("%.2f", rng.nextDouble() * 999.0 + 1.0));
                w.write(',');
                w.write(String.format("%.2f", rng.nextDouble() * 0.5));
                w.newLine();
            }
        }
        System.out.println("Generated " + rows + " rows to " + filename);
    }
}
