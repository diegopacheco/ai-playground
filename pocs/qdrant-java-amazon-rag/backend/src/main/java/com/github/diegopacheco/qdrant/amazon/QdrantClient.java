package com.github.diegopacheco.qdrant.amazon;

import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;
import tools.jackson.databind.JsonNode;

@Component
class QdrantClient {

    static final String COLLECTION = "products";
    static final int VECTOR_SIZE = 768;

    private final RestClient http;

    QdrantClient(@Value("${qdrant.url}") String url) {
        this.http = RestClient.create(url + "/collections/" + COLLECTION);
    }

    void ensureCollection() {
        if (get("/exists").at("/result/exists").asBoolean()) {
            return;
        }
        put("", Map.of("vectors", Map.of("size", VECTOR_SIZE, "distance", "Cosine")));
        put("/index?wait=true", Map.of("field_name", "category", "field_schema", "keyword"));
        put("/index?wait=true", Map.of("field_name", "price", "field_schema", "float"));
    }

    void upsert(List<Product> products, List<float[]> vectors) {
        List<Map<String, Object>> points = new ArrayList<>();
        for (int i = 0; i < products.size(); i++) {
            points.add(Map.of("id", products.get(i).id(), "vector", vectors.get(i), "payload", payload(products.get(i))));
        }
        put("/points?wait=true", Map.of("points", points));
    }

    List<Hit> query(float[] vector, List<Map<String, Object>> must, int limit) {
        Map<String, Object> body = new LinkedHashMap<>();
        body.put("query", vector);
        body.put("limit", limit);
        body.put("with_payload", true);
        if (!must.isEmpty()) {
            body.put("filter", Map.of("must", must));
        }
        List<Hit> hits = new ArrayList<>();
        for (JsonNode point : post("/points/query", body).at("/result/points")) {
            hits.add(hit(point));
        }
        return hits;
    }

    List<Category> categories() {
        List<Category> categories = new ArrayList<>();
        for (JsonNode facet : post("/facet", Map.of("key", "category", "limit", 50)).at("/result/hits")) {
            categories.add(new Category(facet.path("value").asString(), facet.path("count").asLong()));
        }
        return categories;
    }

    long count() {
        return post("/points/count", Map.of("exact", true)).at("/result/count").asLong();
    }

    private static Map<String, Object> payload(Product product) {
        Map<String, Object> payload = new LinkedHashMap<>();
        payload.put("name", product.name());
        payload.put("category", product.category());
        payload.put("category_path", product.categoryPath());
        if (product.price() != null) {
            payload.put("price", product.price());
        }
        payload.put("about", product.about());
        payload.put("image", product.image());
        payload.put("url", product.url());
        return payload;
    }

    private static Hit hit(JsonNode point) {
        JsonNode payload = point.path("payload");
        JsonNode price = payload.path("price");
        return new Hit(
                point.path("id").asString(),
                point.path("score").asDouble(),
                payload.path("name").asString(""),
                payload.path("category").asString(""),
                price.isNumber() ? price.asDouble() : null,
                payload.path("about").asString(""),
                payload.path("image").asString(""),
                payload.path("url").asString(""));
    }

    private JsonNode get(String path) {
        return http.get().uri(path).retrieve().body(JsonNode.class);
    }

    private JsonNode put(String path, Object body) {
        return http.put().uri(path).body(body).retrieve().body(JsonNode.class);
    }

    private JsonNode post(String path, Object body) {
        return http.post().uri(path).body(body).retrieve().body(JsonNode.class);
    }
}
