package com.github.diegopacheco.qdrant.amazon;

import java.nio.charset.StandardCharsets;
import java.util.Arrays;
import java.util.Map;
import java.util.Optional;
import java.util.UUID;
import java.util.regex.Matcher;
import java.util.regex.Pattern;
import java.util.stream.Collectors;
import org.springframework.web.util.HtmlUtils;

final class ProductCleaner {

    static final String UNCATEGORIZED = "Uncategorized";
    static final int EMBEDDING_TEXT_LIMIT = 1500;
    private static final String BOILERPLATE = "Make sure this fits";
    private static final Pattern DOLLARS = Pattern.compile("\\$\\s*(\\d[\\d,]*(?:\\.\\d+)?)");
    private static final Pattern HEX_ID = Pattern.compile("[0-9a-fA-F]{32}");

    private ProductCleaner() {}

    static String unescape(String raw) {
        return raw == null ? "" : HtmlUtils.htmlUnescape(raw).trim();
    }

    static String topCategory(String rawPath) {
        String top = unescape(rawPath).split("\\|")[0].trim();
        return top.isEmpty() ? UNCATEGORIZED : top;
    }

    static Double price(String raw) {
        if (raw == null) {
            return null;
        }
        Matcher matcher = DOLLARS.matcher(raw);
        return matcher.find() ? Double.valueOf(matcher.group(1).replace(",", "")) : null;
    }

    static String about(String raw) {
        return Arrays.stream(unescape(raw).split("\\|"))
                .map(String::trim)
                .filter(bullet -> !bullet.isEmpty() && !bullet.startsWith(BOILERPLATE))
                .collect(Collectors.joining(" | "));
    }

    static String firstImage(String raw) {
        return raw == null ? "" : raw.split("\\|")[0].trim();
    }

    static String pointId(String uniqId) {
        if (HEX_ID.matcher(uniqId).matches()) {
            String hex = uniqId.toLowerCase();
            return String.join("-", hex.substring(0, 8), hex.substring(8, 12), hex.substring(12, 16), hex.substring(16, 20), hex.substring(20));
        }
        return UUID.nameUUIDFromBytes(uniqId.getBytes(StandardCharsets.UTF_8)).toString();
    }

    static String embeddingText(Product product) {
        String text = product.name() + "\n" + product.category() + "\n" + product.about();
        return text.length() > EMBEDDING_TEXT_LIMIT ? text.substring(0, EMBEDDING_TEXT_LIMIT) : text;
    }

    static Optional<Product> fromRow(Map<String, Object> row) {
        String name = unescape(text(row, "Product Name"));
        String uniqId = text(row, "Uniq Id");
        if (name.isEmpty() || uniqId == null || uniqId.isBlank()) {
            return Optional.empty();
        }
        String categoryPath = unescape(text(row, "Category"));
        return Optional.of(new Product(
                pointId(uniqId.trim()),
                name,
                topCategory(categoryPath),
                categoryPath,
                price(text(row, "Selling Price")),
                about(text(row, "About Product")),
                firstImage(text(row, "Image")),
                unescape(text(row, "Product Url"))));
    }

    private static String text(Map<String, Object> row, String column) {
        Object value = row.get(column);
        return value == null ? null : value.toString();
    }
}
