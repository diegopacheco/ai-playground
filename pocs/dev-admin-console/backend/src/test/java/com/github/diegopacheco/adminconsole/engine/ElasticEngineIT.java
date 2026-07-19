package com.github.diegopacheco.adminconsole.engine;

import static org.assertj.core.api.Assertions.assertThat;
import static org.assertj.core.api.Assertions.assertThatThrownBy;

import com.github.diegopacheco.adminconsole.engine.elastic.ElasticEngine;
import com.github.diegopacheco.adminconsole.project.ConnectionConfig;
import com.github.diegopacheco.adminconsole.project.ConnectionKind;
import java.util.HashSet;
import java.util.List;
import java.util.Set;
import org.junit.jupiter.api.Tag;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;

@Tag("integration-test")
@SpringBootTest
class ElasticEngineIT {
    @Autowired
    private ElasticEngine elastic;

    private final ConnectionConfig config = new ConnectionConfig(9301L, 1L, "demo-elastic",
            ConnectionKind.ELASTICSEARCH, "localhost", 9200, null, null, null, null, null, null, "tester");

    @Test
    void listsIndicesWithTheirMappingFieldsIncludingNestedObjects() {
        List<SchemaNode> indices = elastic.schema(config);
        assertThat(indices).extracting(SchemaNode::name).contains("products");
        SchemaNode products = indices.stream().filter(node -> node.name().equals("products")).findFirst().orElseThrow();
        assertThat(products.children()).extracting(SchemaNode::name)
                .contains("sku", "name", "price_cents", "in_stock", "category", "supplier");
        SchemaNode supplier = products.children().stream().filter(node -> node.name().equals("supplier"))
                .findFirst().orElseThrow();
        assertThat(supplier.children()).extracting(SchemaNode::name).contains("name", "country");
    }

    @Test
    void reportsFieldTypesSoTheTreeCanShowWhatEachFieldIs() {
        SchemaNode products = elastic.schema(config).stream()
                .filter(node -> node.name().equals("products")).findFirst().orElseThrow();
        assertThat(products.children()).extracting(SchemaNode::detail).contains("keyword", "text", "integer", "boolean");
    }

    @Test
    void runsASearchAndFlattensHitsIntoColumns() {
        QueryResult result = elastic.query(config, "GET /products/_search", PageRequest.first(10));
        assertThat(result.columns()).contains("_id", "sku", "name", "price_cents");
        assertThat(result.rows()).hasSize(10);
    }

    @Test
    void acceptsAQueryDslBodyLikeKibanaDevTools() {
        QueryResult result = elastic.query(config,
                "GET /products/_search {\"query\":{\"term\":{\"category\":\"cat-1\"}}}", PageRequest.first(5));
        assertThat(result.rows()).hasSize(5);
        assertThat(result.rows()).allMatch(row -> row.get("category").equals("cat-1"));
    }

    @Test
    void pagesWithSearchAfterWithoutRepeatingDocuments() {
        QueryResult first = elastic.query(config, "GET /products/_search", PageRequest.first(50));
        assertThat(first.hasMore()).isTrue();
        QueryResult second = elastic.query(config, "GET /products/_search",
                new PageRequest(50, first.nextCursor(), 2));
        Set<Object> firstIds = new HashSet<>(first.rows().stream().map(row -> row.get("_id")).toList());
        Set<Object> secondIds = new HashSet<>(second.rows().stream().map(row -> row.get("_id")).toList());
        assertThat(second.rows()).hasSize(50);
        assertThat(firstIds).doesNotContainAnyElementsOf(secondIds);
    }

    @Test
    void pagesPastTheResultWindowWhereFromAndSizeWouldFail() {
        int window = 200;
        QueryResult result = elastic.query(config, "GET /products/_search", PageRequest.first(50));
        int seen = result.rows().size();
        for (int page = 2; page <= 20 && result.hasMore(); page++) {
            result = elastic.query(config, "GET /products/_search", new PageRequest(50, result.nextCursor(), page));
            seen += result.rows().size();
        }
        assertThat(seen).isGreaterThan(window);
    }

    @Test
    void readsCatAndCountEndpoints() {
        assertThat(elastic.query(config, "GET /_cat/indices?format=json", PageRequest.first(20)).rows())
                .isNotEmpty();
        assertThat(elastic.query(config, "GET /products/_count", PageRequest.first(10)).rows())
                .anyMatch(row -> "count".equals(row.get("field")) && "1000".equals(row.get("value")));
    }

    @Test
    void rejectsEveryNonReadVerb() {
        assertThatThrownBy(() -> elastic.query(config, "DELETE /products", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> elastic.query(config, "PUT /products/_doc/1 {}", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> elastic.query(config, "POST /products/_search", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void rejectsMutatingEndpointsEvenWhenDisguisedAsAGetBecauseSeveralOfThemAcceptGet() {
        assertThatThrownBy(() -> elastic.query(config,
                "GET /products/_delete_by_query {\"query\":{\"match_all\":{}}}", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> elastic.query(config, "GET /_bulk", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
        assertThatThrownBy(() -> elastic.query(config, "GET /products/_update_by_query", PageRequest.first(10)))
                .isInstanceOf(ReadOnlyViolation.class);
    }

    @Test
    void leavesTheIndexIntactAfterEveryRejectedWriteAttempt() {
        assertThat(elastic.query(config, "GET /products/_count", PageRequest.first(10)).rows())
                .anyMatch(row -> "count".equals(row.get("field")) && "1000".equals(row.get("value")));
    }
}
