package com.github.diegopacheco.adminconsole.engine.elastic;

import com.github.diegopacheco.adminconsole.engine.Engine;
import com.github.diegopacheco.adminconsole.engine.PageRequest;
import com.github.diegopacheco.adminconsole.engine.QueryResult;
import com.github.diegopacheco.adminconsole.engine.SchemaNode;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.net.URI;
import java.net.http.HttpClient;
import java.net.http.HttpRequest;
import java.net.http.HttpResponse;
import java.nio.charset.StandardCharsets;
import java.time.Duration;
import java.util.ArrayList;
import java.util.Base64;
import java.util.LinkedHashMap;
import java.util.LinkedHashSet;
import java.util.List;
import java.util.Map;
import java.util.Set;
import org.springframework.stereotype.Component;
import tools.jackson.databind.JsonNode;
import tools.jackson.databind.ObjectMapper;

@Component
public class ElasticEngine implements Engine {
    private final ElasticCommandParser parser;
    private final ElasticEndpointGuard guard;
    private final ObjectMapper mapper = new ObjectMapper();
    private final HttpClient client = HttpClient.newBuilder().connectTimeout(Duration.ofSeconds(5)).build();

    public ElasticEngine(ElasticCommandParser parser, ElasticEndpointGuard guard) {
        this.parser = parser;
        this.guard = guard;
    }

    @Override
    public ConnectionKind kind() {
        return ConnectionKind.ELASTICSEARCH;
    }

    @Override
    public void assertReadOnly(String statement) {
        guard.assertReadOnly(parser.parse(statement));
    }

    @Override
    public List<SchemaNode> schema(ConnectionConfig config) {
        JsonNode mappings = mapper.readTree(send(config, new ElasticRequest("GET", "/_all/_mapping", null)));
        List<SchemaNode> indices = new ArrayList<>();
        mappings.propertyNames().forEach(index -> {
            if (index.startsWith(".")) {
                return;
            }
            JsonNode properties = mappings.get(index).path("mappings").path("properties");
            indices.add(new SchemaNode(index, "index", fieldCount(properties) + " fields", fields(properties)));
        });
        indices.sort((left, right) -> left.name().compareTo(right.name()));
        return indices;
    }

    @Override
    public QueryResult query(ConnectionConfig config, String statement, PageRequest page) {
        ElasticRequest request = parser.parse(statement);
        guard.assertReadOnly(request);
        if (request.path().contains("_search")) {
            return search(config, request, page);
        }
        String response = send(config, request);
        return raw(response, page);
    }

    private QueryResult search(ConnectionConfig config, ElasticRequest request, PageRequest page) {
        Map<String, Object> body = request.body() == null
                ? new LinkedHashMap<>()
                : mapper.readValue(request.body(), LinkedHashMap.class);
        body.put("size", page.size());
        if (!body.containsKey("sort")) {
            body.put("sort", List.of(Map.of("_doc", "asc")));
        }
        if (!page.isFirst()) {
            body.put("search_after", mapper.readValue(
                    new String(Base64.getDecoder().decode(page.cursor()), StandardCharsets.UTF_8), List.class));
        }
        JsonNode response = mapper.readTree(send(config,
                new ElasticRequest("GET", request.path(), mapper.writeValueAsString(body))));
        JsonNode hits = response.path("hits").path("hits");
        Set<String> columns = new LinkedHashSet<>(List.of("_id", "_score"));
        List<Map<String, Object>> rows = new ArrayList<>();
        JsonNode lastSort = null;
        for (JsonNode hit : hits) {
            Map<String, Object> row = new LinkedHashMap<>();
            row.put("_id", hit.path("_id").asString());
            row.put("_score", hit.path("_score").isNull() ? null : hit.path("_score").asString());
            JsonNode source = hit.path("_source");
            source.propertyNames().forEach(field -> {
                columns.add(field);
                row.put(field, (source.get(field).isObject() || source.get(field).isArray())
                        ? source.get(field).toString() : source.get(field).asString());
            });
            rows.add(row);
            if (hit.has("sort")) {
                lastSort = hit.get("sort");
            }
        }
        boolean hasMore = rows.size() == page.size() && lastSort != null;
        String nextCursor = hasMore
                ? Base64.getEncoder().encodeToString(lastSort.toString().getBytes(StandardCharsets.UTF_8))
                : null;
        return QueryResult.of(new ArrayList<>(columns), rows, page.pageNumber(), nextCursor, hasMore);
    }

    private QueryResult raw(String response, PageRequest page) {
        JsonNode node = mapper.readTree(response);
        List<Map<String, Object>> rows = new ArrayList<>();
        Set<String> columns = new LinkedHashSet<>();
        if (node.isArray()) {
            for (JsonNode element : node) {
                Map<String, Object> row = new LinkedHashMap<>();
                element.propertyNames().forEach(field -> {
                    columns.add(field);
                    row.put(field, element.get(field).asString());
                });
                rows.add(row);
            }
        } else {
            columns.add("field");
            columns.add("value");
            node.propertyNames().forEach(field -> {
                Map<String, Object> row = new LinkedHashMap<>();
                row.put("field", field);
                row.put("value", (node.get(field).isObject() || node.get(field).isArray())
                        ? node.get(field).toString() : node.get(field).asString());
                rows.add(row);
            });
        }
        int offset = page.isFirst() ? 0 : Integer.parseInt(page.cursor());
        List<Map<String, Object>> window = rows.stream().skip(offset).limit(page.size()).toList();
        boolean hasMore = rows.size() > offset + window.size();
        return QueryResult.of(new ArrayList<>(columns), window, page.pageNumber(),
                hasMore ? String.valueOf(offset + window.size()) : null, hasMore);
    }

    private List<SchemaNode> fields(JsonNode properties) {
        List<SchemaNode> nodes = new ArrayList<>();
        properties.propertyNames().forEach(field -> {
            JsonNode definition = properties.get(field);
            JsonNode nested = definition.path("properties");
            String type = definition.path("type").asString("object");
            nodes.add(new SchemaNode(field, "field", type, nested.isObject() ? fields(nested) : List.of()));
        });
        return nodes;
    }

    private int fieldCount(JsonNode properties) {
        return properties.propertyNames().size();
    }

    private String send(ConnectionConfig config, ElasticRequest request) {
        try {
            String url = "http://" + config.host() + ":" + config.port() + request.path();
            HttpRequest.Builder builder = HttpRequest.newBuilder(URI.create(url))
                    .timeout(Duration.ofSeconds(30))
                    .header("Content-Type", "application/json");
            if (config.username() != null && !config.username().isBlank()) {
                String credentials = config.username() + ":" + (config.password() == null ? "" : config.password());
                builder.header("Authorization", "Basic "
                        + Base64.getEncoder().encodeToString(credentials.getBytes(StandardCharsets.UTF_8)));
            }
            builder.method(request.method(), request.body() == null
                    ? HttpRequest.BodyPublishers.noBody()
                    : HttpRequest.BodyPublishers.ofString(request.body()));
            HttpResponse<String> response = client.send(builder.build(), HttpResponse.BodyHandlers.ofString());
            if (response.statusCode() >= 400) {
                throw new IllegalStateException("elasticsearch returned " + response.statusCode() + ": " + response.body());
            }
            return response.body();
        } catch (InterruptedException error) {
            Thread.currentThread().interrupt();
            throw new IllegalStateException("elasticsearch request was interrupted", error);
        } catch (java.io.IOException error) {
            throw new IllegalStateException("elasticsearch request failed: " + error.getMessage(), error);
        }
    }
}
