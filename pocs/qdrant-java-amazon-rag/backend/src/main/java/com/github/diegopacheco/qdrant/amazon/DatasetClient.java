package com.github.diegopacheco.qdrant.amazon;

import java.util.List;
import java.util.Map;
import org.springframework.core.ParameterizedTypeReference;
import org.springframework.stereotype.Component;
import org.springframework.web.client.RestClient;

@Component
class DatasetClient {

    static final int PAGE_SIZE = 100;
    private static final String DATASET = "bprateek/amazon_product_description";

    private final RestClient http = RestClient.create("https://datasets-server.huggingface.co");

    record Page(List<Entry> rows) {}

    record Entry(Map<String, Object> row) {}

    List<Map<String, Object>> rows(int offset) {
        Page page = http.get()
                .uri("/rows?dataset={dataset}&config=default&split=train&offset={offset}&length={length}", DATASET, offset, PAGE_SIZE)
                .retrieve()
                .body(new ParameterizedTypeReference<Page>() {});
        return page.rows().stream().map(Entry::row).toList();
    }
}
