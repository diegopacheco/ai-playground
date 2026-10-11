package com.github.diegopacheco.qdrant.amazon;

import java.util.Comparator;
import java.util.List;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.bind.annotation.GetMapping;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestBody;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.ResponseStatus;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.client.RestClientException;

@RestController
@RequestMapping("/api")
class ApiController {

    static final int DEFAULT_LIMIT = 10;
    private static final Logger LOG = LoggerFactory.getLogger(ApiController.class);

    private final QdrantClient qdrant;
    private final SearchService search;
    private final RagService rag;
    private final IngestService ingest;
    private final int productLimit;

    ApiController(QdrantClient qdrant, SearchService search, RagService rag, IngestService ingest,
                  @Value("${ingest.product-limit}") int productLimit) {
        this.qdrant = qdrant;
        this.search = search;
        this.rag = rag;
        this.ingest = ingest;
        this.productLimit = productLimit;
    }

    @GetMapping("/stats")
    Stats stats() {
        return new Stats(qdrant.count(), QdrantClient.COLLECTION);
    }

    @GetMapping("/categories")
    List<Category> categories() {
        return qdrant.categories().stream().sorted(Comparator.comparingLong(Category::count).reversed()).toList();
    }

    @PostMapping("/search")
    SearchResponse search(@RequestBody SearchRequest request) {
        requireText(request.query(), "query");
        int limit = request.limit() == null ? DEFAULT_LIMIT : request.limit();
        return new SearchResponse(search.search(request.query(), limit, request.category(), request.minPrice(), request.maxPrice()));
    }

    @PostMapping("/ask")
    AskResponse ask(@RequestBody AskRequest request) {
        requireText(request.question(), "question");
        return rag.ask(request.question(), request.category(), request.minPrice(), request.maxPrice());
    }

    @PostMapping("/ingest")
    IngestResponse ingest(@RequestBody(required = false) IngestRequest request) {
        return ingest.ingest(request == null || request.limit() == null ? productLimit : request.limit());
    }

    @ExceptionHandler(IllegalArgumentException.class)
    @ResponseStatus(HttpStatus.BAD_REQUEST)
    ErrorResponse badRequest(IllegalArgumentException error) {
        return new ErrorResponse(error.getMessage());
    }

    @ExceptionHandler(RestClientException.class)
    @ResponseStatus(HttpStatus.BAD_GATEWAY)
    ErrorResponse badGateway(RestClientException error) {
        LOG.error("upstream call failed", error);
        return new ErrorResponse(error.getMessage());
    }

    private static void requireText(String value, String field) {
        if (value == null || value.isBlank()) {
            throw new IllegalArgumentException(field + " must not be blank");
        }
    }
}
