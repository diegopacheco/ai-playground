package com.github.diegopacheco.qdrant.amazon;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.stereotype.Service;

@Service
class SearchService {

    static final String QUERY_PREFIX = "search_query: ";

    private final OllamaClient ollama;
    private final QdrantClient qdrant;

    SearchService(OllamaClient ollama, QdrantClient qdrant) {
        this.ollama = ollama;
        this.qdrant = qdrant;
    }

    List<Hit> search(String query, int limit, String category, Double minPrice, Double maxPrice) {
        float[] vector = ollama.embed(List.of(QUERY_PREFIX + query)).get(0);
        return qdrant.query(vector, filter(category, minPrice, maxPrice), limit);
    }

    static List<Map<String, Object>> filter(String category, Double minPrice, Double maxPrice) {
        List<Map<String, Object>> must = new ArrayList<>();
        if (category != null && !category.isBlank()) {
            must.add(Map.of("key", "category", "match", Map.of("value", category)));
        }
        Map<String, Object> range = new LinkedHashMap<>();
        if (minPrice != null) {
            range.put("gte", minPrice);
        }
        if (maxPrice != null) {
            range.put("lte", maxPrice);
        }
        if (!range.isEmpty()) {
            must.add(Map.of("key", "price", "range", range));
        }
        return must;
    }
}
