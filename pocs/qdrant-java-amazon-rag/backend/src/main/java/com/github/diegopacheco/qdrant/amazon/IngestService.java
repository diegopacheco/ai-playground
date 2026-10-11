package com.github.diegopacheco.qdrant.amazon;

import java.util.ArrayList;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import org.springframework.stereotype.Service;

@Service
class IngestService {

    static final int EMBED_BATCH = 32;
    static final String DOCUMENT_PREFIX = "search_document: ";

    private final DatasetClient dataset;
    private final OllamaClient ollama;
    private final QdrantClient qdrant;

    IngestService(DatasetClient dataset, OllamaClient ollama, QdrantClient qdrant) {
        this.dataset = dataset;
        this.ollama = ollama;
        this.qdrant = qdrant;
    }

    IngestResponse ingest(int limit) {
        qdrant.ensureCollection();
        int indexed = 0;
        int skipped = 0;
        int offset = 0;
        while (indexed < limit) {
            List<Map<String, Object>> rows = dataset.rows(offset);
            if (rows.isEmpty()) {
                break;
            }
            offset += rows.size();
            List<Product> products = new ArrayList<>();
            for (Map<String, Object> row : rows) {
                if (indexed + products.size() == limit) {
                    break;
                }
                Optional<Product> product = ProductCleaner.fromRow(row);
                if (product.isPresent()) {
                    products.add(product.get());
                } else {
                    skipped++;
                }
            }
            for (int start = 0; start < products.size(); start += EMBED_BATCH) {
                index(products.subList(start, Math.min(start + EMBED_BATCH, products.size())));
            }
            indexed += products.size();
        }
        return new IngestResponse(indexed, skipped);
    }

    private void index(List<Product> batch) {
        List<String> texts = batch.stream().map(product -> DOCUMENT_PREFIX + ProductCleaner.embeddingText(product)).toList();
        qdrant.upsert(batch, ollama.embed(texts));
    }
}
